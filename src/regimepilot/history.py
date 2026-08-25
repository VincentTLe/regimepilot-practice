"""Read-only market-data reads for one FeaturePacket.

SAFETY: this module is read-only by design, exactly like Phase 1's smoke test
and Phase 2A's observer. It contains no function that submits, cancels or
replaces an order, and none that closes or exercises a position. Do not add one
here.

This is the only Phase 2B module that knows Alpaca exists. It fetches, it
normalizes, and it hands plain ``OhlcvBar`` values to ``features``, which does
the arithmetic without ever seeing an SDK object.

Two failure modes are kept strictly apart, as in Phase 2A:

* A call succeeds but carries no data -> an empty list, or ``None``. A session
  that has not started yet is a fact, and the features derived from it are
  null.
* A call fails -> ``HistoryError``. No bar is ever invented to paper over a
  broken request, so an empty FeaturePacket always means "the market had
  nothing to say", never "the API was down".

Every historical request names ``DataFeed.IEX`` explicitly. Alpaca's default is
"the best feed your subscription allows", which would silently change the
meaning of a feature the day a subscription changes.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from regimepilot.config import ConfigError, Settings, load_settings
from regimepilot.features import (
    DATA_FEED_IEX,
    FeaturePacket,
    build_feature_packet,
    format_summary,
    session_bounds,
    session_date_of,
    to_utc,
)
from regimepilot.models import UNDERLYING_SYMBOL, OhlcvBar

# Reused rather than reimplemented, so a bar is normalized identically in both
# phases and client construction stays paper-only in one place.
from regimepilot.observer import normalize_bar
from regimepilot.smoke_test import build_clients

UNDERLYING = UNDERLYING_SYMBOL

# Never left to the account default: this phase is defined on IEX.
HISTORICAL_FEED = DataFeed.IEX

# Calendar days of daily bars to request. Wide enough that a long weekend plus
# a holiday still leaves a previous regular session to find.
DAILY_BAR_LOOKBACK_DAYS = 10

__all__ = [
    "HistoryError",
    "fetch_latest_quote",
    "fetch_market_clock",
    "fetch_recent_daily_bars",
    "fetch_session_minute_bars",
    "format_summary",
    "observe_features",
    "main",
]


class HistoryError(RuntimeError):
    """A read-only market-data request could not be completed.

    The message names the step that failed and the exception type, never the
    upstream text: an HTTP client's message can quote the request it made,
    which would put a key in a log.
    """


def _guarded(label: str, call: Callable[[], Any]) -> Any:
    """Run one request, converting any failure into a credential-safe error.

    ``from None`` drops the upstream exception instead of chaining it, so no
    traceback printed from a HistoryError can echo an outbound request.
    """
    try:
        return call()
    except Exception as error:  # noqa: BLE001 - deliberately uniform
        raise HistoryError(f"failed to read {label}: {type(error).__name__}") from None


def _as_float(value: Any) -> float | None:
    """Coerce a scalar to float, or ``None`` if it is not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows(response: Any, symbol: str) -> list[Any]:
    """Pull one symbol's list out of a ``{symbol: [...]}`` reply."""
    data = response if isinstance(response, dict) else getattr(response, "data", None)
    return list((data or {}).get(symbol) or [])


def _row(response: Any, symbol: str) -> Any:
    """Pull one symbol's single model out of a ``{symbol: model}`` reply."""
    data = response if isinstance(response, dict) else getattr(response, "data", None)
    return (data or {}).get(symbol)


def _normalized_bars(response: Any, symbol: str) -> list[OhlcvBar]:
    bars = (normalize_bar(row) for row in _rows(response, symbol))
    return [bar for bar in bars if bar is not None]


def fetch_session_minute_bars(
    data_client: Any,
    *,
    observed_at: datetime | None = None,
    symbol: str = UNDERLYING,
) -> list[OhlcvBar]:
    """One-minute bars for the regular session that contains ``observed_at``.

    The window is exactly 09:30-16:00 New York of that session, so a reply can
    never contain the previous trading day even before ``features`` filters it.
    Both ends are inclusive at Alpaca, so the 16:00 bar and the currently
    forming minute can both come back; ``features`` drops them.

    A session that has not opened yet simply returns nothing.
    """
    observed_at = to_utc(observed_at) if observed_at else datetime.now(timezone.utc)
    opens, closes = session_bounds(session_date_of(observed_at))

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=opens,
        end=closes,
        feed=HISTORICAL_FEED,
    )
    response = _guarded(
        f"{symbol} minute bars", lambda: data_client.get_stock_bars(request)
    )
    return _normalized_bars(response, symbol)


def fetch_recent_daily_bars(
    data_client: Any,
    *,
    observed_at: datetime | None = None,
    symbol: str = UNDERLYING,
) -> list[OhlcvBar]:
    """Recent daily bars, for the previous regular session's official close.

    A daily bar's close excludes extended-hours trades, so it is the regular
    session close the overnight gap is defined against.
    """
    end = to_utc(observed_at) if observed_at else datetime.now(timezone.utc)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
        start=end - timedelta(days=DAILY_BAR_LOOKBACK_DAYS),
        end=end,
        feed=HISTORICAL_FEED,
    )
    response = _guarded(
        f"{symbol} daily bars", lambda: data_client.get_stock_bars(request)
    )
    return _normalized_bars(response, symbol)


def fetch_latest_quote(
    data_client: Any, *, symbol: str = UNDERLYING
) -> tuple[float | None, float | None]:
    """Top of book as ``(bid, ask)``. A quiet feed gives ``(None, None)``."""
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=HISTORICAL_FEED)
    response = _guarded(
        f"{symbol} latest quote", lambda: data_client.get_stock_latest_quote(request)
    )
    quote = _row(response, symbol)
    return (
        _as_float(getattr(quote, "bid_price", None)),
        _as_float(getattr(quote, "ask_price", None)),
    )


class MarketClock(NamedTuple):
    """The two clock facts a FeaturePacket needs, from one clock request.

    ``next_close`` is this session's close while the market is open, which is
    the only time the feature layer uses it. Reading it here rather than
    assuming 16:00 is what keeps a half day from over-reporting the time left.
    """

    is_open: bool | None
    next_close: datetime | None


def fetch_market_clock(trading_client: Any) -> MarketClock:
    """Read the equity market clock once, for both openness and the close time."""
    clock = _guarded("market clock", trading_client.get_clock)
    is_open = getattr(clock, "is_open", None)
    next_close = getattr(clock, "next_close", None)
    return MarketClock(
        is_open=None if is_open is None else bool(is_open),
        next_close=to_utc(next_close) if isinstance(next_close, datetime) else None,
    )


class _MarketInputs(NamedTuple):
    """Everything one observation reads, before any arithmetic touches it."""

    market_is_open: bool | None
    session_close_at: datetime | None
    minute_bars: list[OhlcvBar]
    daily_bars: list[OhlcvBar]
    bid: float | None
    ask: float | None


def _read_inputs(
    trading_client: Any, data_client: Any, observed_at: datetime, symbol: str
) -> _MarketInputs:
    """Every read-only request one observation needs, made exactly once each."""
    clock = fetch_market_clock(trading_client)
    minute_bars = fetch_session_minute_bars(
        data_client, observed_at=observed_at, symbol=symbol
    )
    daily_bars = fetch_recent_daily_bars(
        data_client, observed_at=observed_at, symbol=symbol
    )
    bid, ask = fetch_latest_quote(data_client, symbol=symbol)
    return _MarketInputs(
        clock.is_open, clock.next_close, minute_bars, daily_bars, bid, ask
    )


def observe_features(
    trading_client: Any,
    data_client: Any,
    *,
    now: datetime | None = None,
    symbol: str = UNDERLYING,
) -> FeaturePacket:
    """Take one read-only observation and derive a FeaturePacket from it.

    Clients are injected so unit tests can run this without a network call.
    Raises ``HistoryError`` if any single request fails; a partial packet is
    never returned.
    """
    observed_at = to_utc(now) if now else datetime.now(timezone.utc)
    inputs = _read_inputs(trading_client, data_client, observed_at, symbol)

    return build_feature_packet(
        observed_at=observed_at,
        minute_bars=inputs.minute_bars,
        daily_bars=inputs.daily_bars,
        bid=inputs.bid,
        ask=inputs.ask,
        market_is_open=inputs.market_is_open,
        session_close_at=inputs.session_close_at,
        symbol=symbol,
        data_feed=DATA_FEED_IEX,
    )


def _bar_coverage(bars: Sequence[OhlcvBar]) -> str:
    """One line saying how much session history the request actually returned."""
    if not bars:
        return "no bars returned for this session"
    stamps = sorted(bar.timestamp for bar in bars if bar.timestamp is not None)
    if not stamps:
        return f"{len(bars)} bars, none timestamped"
    return (
        f"{len(bars)} bars returned, {stamps[0].strftime('%H:%M')}Z"
        f" -> {stamps[-1].strftime('%H:%M')}Z"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print a compact FeaturePacket summary, or the packet itself with ``--json``."""
    arguments = list(sys.argv[1:] if argv is None else argv)

    try:
        settings: Settings = load_settings()
        trading_client, data_client = build_clients(settings)
    except ConfigError as error:
        # ConfigError messages are built by us and never contain a credential.
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    observed_at = datetime.now(timezone.utc)
    try:
        inputs = _read_inputs(trading_client, data_client, observed_at, UNDERLYING)
    except HistoryError as error:
        print(f"history read failed: {error}", file=sys.stderr)
        return 1

    packet = build_feature_packet(
        observed_at=observed_at,
        minute_bars=inputs.minute_bars,
        daily_bars=inputs.daily_bars,
        bid=inputs.bid,
        ask=inputs.ask,
        market_is_open=inputs.market_is_open,
        session_close_at=inputs.session_close_at,
        data_feed=DATA_FEED_IEX,
    )

    if "--json" in arguments:
        print(json.dumps(json.loads(packet.model_dump_json()), indent=2))
    else:
        print(format_summary(packet))
        print(f"  {'source':<20} {_bar_coverage(inputs.minute_bars)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
