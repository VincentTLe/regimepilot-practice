"""Read-only market observer: Alpaca responses in, one ObservationPacket out.

SAFETY: this module is read-only by design, exactly like Phase 1's smoke test.
It contains no function that submits, cancels or replaces an order, and none
that closes or exercises a position. Do not add one here.

Two failure modes are kept strictly apart:

* A call succeeds but carries no data for the symbol -> the matching field is
  ``None``. A quiet feed is a fact worth recording.
* A call fails -> ``ObserverError``. A partial packet is never returned and a
  missing value is never invented, so a consumer can trust that any packet it
  holds was fully observed.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestBarRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import AssetStatus
from alpaca.trading.requests import GetOptionContractsRequest

from regimepilot.config import ConfigError, Settings, load_settings
from regimepilot.models import (
    UNDERLYING_SYMBOL,
    AccountSnapshot,
    MarketState,
    ObservationPacket,
    OhlcvBar,
    OptionContractSummary,
    OptionUniverse,
    UnderlyingSnapshot,
)

# Reused rather than reimplemented, so masking and client construction behave
# identically in both phases.
from regimepilot.smoke_test import build_clients, mask_account_id

UNDERLYING = UNDERLYING_SYMBOL

# Option expirations are US market calendar dates, so days-to-expiration has to
# be counted from the market's date rather than from UTC's. Between 00:00 UTC
# and New York midnight the two disagree: UTC is already on the next day while
# the options market is still on the previous one. Declared here rather than
# imported from a later phase, so this module stays free-standing.
MARKET_TIMEZONE = ZoneInfo("America/New_York")

# Calendar days of daily bars to request. Only the last two are kept; the span
# is wide enough that a weekend plus a holiday still leaves two sessions.
DAILY_BAR_LOOKBACK_DAYS = 10

# Inherited unchanged from Phase 1's smoke test rather than chosen here.
MIN_DAYS_TO_EXPIRATION = 3
MAX_DAYS_TO_EXPIRATION = 14

# SPY expires daily, so one window holds thousands of contracts. Page to
# exhaustion, because a truncated page would make contract_count and the
# expiration bounds describe a slice while claiming to describe the window.
# Running past the cap is therefore a failed observation, not a smaller one.
OPTION_CONTRACT_PAGE_LIMIT = 10_000
MAX_OPTION_CONTRACT_PAGES = 20


class ObserverError(RuntimeError):
    """An observation could not be completed.

    The message names the step that failed and the exception type, never the
    upstream text: an HTTP client's message can quote the request it made.
    """


def _guarded(label: str, call: Callable[[], Any]) -> Any:
    """Run one API call, converting any failure into a credential-safe error.

    ``from None`` drops the upstream exception instead of chaining it, so no
    traceback printed from an ObserverError can echo an outbound request.
    """
    try:
        return call()
    except ObserverError:
        # Already ours, and already built without upstream text. Re-wrapping it
        # would replace a specific, safe message with a vaguer one.
        raise
    except Exception as error:  # noqa: BLE001 - deliberately uniform
        raise ObserverError(f"failed to observe {label}: {type(error).__name__}") from None


def _as_float(value: Any) -> float | None:
    """Coerce an Alpaca scalar to float. Alpaca sends money as strings."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    """Unwrap an Alpaca enum (``ContractType.CALL`` -> ``"call"``)."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _field(source: Any, name: str) -> Any:
    return getattr(source, name, None)


def normalize_bar(bar: Any) -> OhlcvBar | None:
    """Convert one Alpaca bar. ``None`` in, ``None`` out."""
    if bar is None:
        return None
    return OhlcvBar(
        timestamp=_field(bar, "timestamp"),
        open=_as_float(_field(bar, "open")),
        high=_as_float(_field(bar, "high")),
        low=_as_float(_field(bar, "low")),
        close=_as_float(_field(bar, "close")),
        volume=_as_float(_field(bar, "volume")),
    )


def normalize_market(clock: Any) -> MarketState:
    is_open = _field(clock, "is_open")
    return MarketState(
        is_open=None if is_open is None else bool(is_open),
        next_open=_field(clock, "next_open"),
        next_close=_field(clock, "next_close"),
    )


def normalize_account(account: Any) -> AccountSnapshot:
    """Convert an Alpaca account, keeping only the masked id."""
    return AccountSnapshot(
        account_id_masked=mask_account_id(_field(account, "id")),
        equity=_as_float(_field(account, "equity")),
        cash=_as_float(_field(account, "cash")),
        buying_power=_as_float(_field(account, "buying_power")),
        options_buying_power=_as_float(_field(account, "options_buying_power")),
        options_trading_level=_as_int(_field(account, "options_trading_level")),
    )


def normalize_contract(contract: Any) -> OptionContractSummary:
    tradable = _field(contract, "tradable")
    return OptionContractSummary(
        symbol=str(_field(contract, "symbol") or ""),
        option_type=_as_text(_field(contract, "type")),
        strike_price=_as_float(_field(contract, "strike_price")),
        expiration_date=_field(contract, "expiration_date"),
        status=_as_text(_field(contract, "status")),
        tradable=None if tradable is None else bool(tradable),
    )


def normalize_option_universe(contracts: Iterable[Any]) -> OptionUniverse:
    """Summarize the contracts observed. An empty list is a valid universe."""
    summaries = [normalize_contract(contract) for contract in contracts]
    expirations = sorted(s.expiration_date for s in summaries if s.expiration_date is not None)
    return OptionUniverse(
        contract_count=len(summaries),
        earliest_expiration=expirations[0] if expirations else None,
        latest_expiration=expirations[-1] if expirations else None,
        contracts=tuple(summaries),
    )


def normalize_underlying(
    *,
    trade: Any = None,
    quote: Any = None,
    minute_bar: Any = None,
    daily_bars: Sequence[Any] = (),
) -> UnderlyingSnapshot:
    """Assemble the underlying view. Each input is independently optional."""
    dated = [bar for bar in daily_bars if _field(bar, "timestamp") is not None]
    ordered = sorted(dated, key=lambda bar: bar.timestamp)
    daily = ordered[-1] if ordered else None
    previous_daily = ordered[-2] if len(ordered) >= 2 else None

    return UnderlyingSnapshot(
        symbol=UNDERLYING,
        latest_trade_price=_as_float(_field(trade, "price")),
        latest_trade_timestamp=_field(trade, "timestamp"),
        bid_price=_as_float(_field(quote, "bid_price")),
        ask_price=_as_float(_field(quote, "ask_price")),
        quote_timestamp=_field(quote, "timestamp"),
        minute_bar=normalize_bar(minute_bar),
        daily_bar=normalize_bar(daily),
        previous_daily_bar=normalize_bar(previous_daily),
    )


def _latest(response: Any) -> Any:
    """Pull our symbol out of Alpaca's ``{symbol: model}`` latest-data reply."""
    if not isinstance(response, dict):
        response = _field(response, "data") or {}
    return response.get(UNDERLYING)


def _fetch_daily_bars(data_client: Any, now: datetime) -> list[Any]:
    request = StockBarsRequest(
        symbol_or_symbols=[UNDERLYING],
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
        start=now - timedelta(days=DAILY_BAR_LOOKBACK_DAYS),
    )
    barset = data_client.get_stock_bars(request)
    data = _field(barset, "data") or {}
    return list(data.get(UNDERLYING) or [])


def _market_date(now: datetime) -> date:
    """The New York calendar date of ``now``. A naive value is taken as UTC.

    Converted through a real timezone, never a fixed offset, so the answer stays
    right on both sides of a daylight saving change.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(MARKET_TIMEZONE).date()


def _fetch_option_contracts(trading_client: Any, now: datetime) -> list[Any]:
    """Page the option-contract endpoint to exhaustion.

    Raises ``ObserverError`` if the window needs more pages than the cap allows,
    because a complete universe is the only kind this observer reports.
    """
    today = _market_date(now)
    contracts: list[Any] = []
    page_token: str | None = None

    for _ in range(MAX_OPTION_CONTRACT_PAGES):
        request = GetOptionContractsRequest(
            underlying_symbols=[UNDERLYING],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=today + timedelta(days=MIN_DAYS_TO_EXPIRATION),
            expiration_date_lte=today + timedelta(days=MAX_DAYS_TO_EXPIRATION),
            limit=OPTION_CONTRACT_PAGE_LIMIT,
            page_token=page_token,
        )
        response = trading_client.get_option_contracts(request)
        contracts.extend(_field(response, "option_contracts") or [])
        page_token = _field(response, "next_page_token")
        if not page_token:
            return contracts

    # More pages remain than the cap allows. Returning what we have would hand
    # back a slice of the window wearing the whole window's name, so the
    # observation fails instead. The message names only our own cap: no page
    # token, no request text, nothing that came from upstream.
    raise ObserverError(
        "failed to observe option contracts: more pages remain after the "
        f"{MAX_OPTION_CONTRACT_PAGES}-page limit, so the universe is incomplete"
    )


def observe(
    trading_client: Any,
    data_client: Any,
    *,
    now: datetime | None = None,
) -> ObservationPacket:
    """Take one complete read-only observation of SPY.

    Clients are injected so unit tests can run this without a network call.
    Raises ``ObserverError`` if any single call fails.
    """
    observed_at = now or datetime.now(timezone.utc)

    clock = _guarded("market clock", trading_client.get_clock)
    account = _guarded("account", trading_client.get_account)

    trade = _latest(
        _guarded(
            "latest trade",
            lambda: data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=UNDERLYING)
            ),
        )
    )
    quote = _latest(
        _guarded(
            "latest quote",
            lambda: data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=UNDERLYING)
            ),
        )
    )
    minute_bar = _latest(
        _guarded(
            "latest minute bar",
            lambda: data_client.get_stock_latest_bar(
                StockLatestBarRequest(symbol_or_symbols=UNDERLYING)
            ),
        )
    )
    daily_bars = _guarded("daily bars", lambda: _fetch_daily_bars(data_client, observed_at))
    contracts = _guarded(
        "option contracts", lambda: _fetch_option_contracts(trading_client, observed_at)
    )

    return ObservationPacket(
        observed_at=observed_at,
        market=normalize_market(clock),
        account=normalize_account(account),
        underlying=normalize_underlying(
            trade=trade, quote=quote, minute_bar=minute_bar, daily_bars=daily_bars
        ),
        option_universe=normalize_option_universe(contracts),
    )


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def _stamp(value: datetime | None) -> str:
    return "-" if value is None else value.strftime("%Y-%m-%d %H:%M:%SZ")


def _bar_line(label: str, bar: OhlcvBar | None) -> str:
    if bar is None:
        return f"  {label:<15} -"
    volume = "-" if bar.volume is None else format(int(bar.volume), ",")
    return (
        f"  {label:<15} O {_number(bar.open)}  H {_number(bar.high)}  "
        f"L {_number(bar.low)}  C {_number(bar.close)}  V {volume}  @ {_stamp(bar.timestamp)}"
    )


def format_summary(packet: ObservationPacket) -> str:
    """A compact human summary. Never lists the individual contracts."""
    market, account = packet.market, packet.account
    underlying, universe = packet.underlying, packet.option_universe

    state = "-" if market.is_open is None else ("OPEN" if market.is_open else "CLOSED")
    expirations = (
        "-"
        if universe.earliest_expiration is None
        else f"{universe.earliest_expiration} -> {universe.latest_expiration}"
    )
    level = "-" if account.options_trading_level is None else account.options_trading_level

    return "\n".join(
        [
            f"RegimePilot observation  {underlying.symbol}  @ {_stamp(packet.observed_at)}",
            f"  {'market':<15} {state}   next open {_stamp(market.next_open)}"
            f"   next close {_stamp(market.next_close)}",
            f"  {'account':<15} {account.account_id_masked}"
            f"   equity {_number(account.equity)}   cash {_number(account.cash)}",
            f"  {'buying power':<15} {_number(account.buying_power)}"
            f"   options {_number(account.options_buying_power)}   level {level}",
            f"  {'last trade':<15} {_number(underlying.latest_trade_price)}"
            f"   @ {_stamp(underlying.latest_trade_timestamp)}",
            f"  {'quote':<15} bid {_number(underlying.bid_price)}"
            f"   ask {_number(underlying.ask_price)}   @ {_stamp(underlying.quote_timestamp)}",
            _bar_line("minute bar", underlying.minute_bar),
            _bar_line("daily bar", underlying.daily_bar),
            _bar_line("previous daily", underlying.previous_daily_bar),
            f"  {'options':<15} {universe.contract_count:,} contracts   {expirations}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print a compact summary, or the full packet with ``--json``."""
    arguments = list(sys.argv[1:] if argv is None else argv)

    try:
        settings: Settings = load_settings()
        trading_client, data_client = build_clients(settings)
    except ConfigError as error:
        # ConfigError messages are built by us and never contain a credential.
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    try:
        packet = observe(trading_client, data_client)
    except ObserverError as error:
        print(f"observation failed: {error}", file=sys.stderr)
        return 1

    if "--json" in arguments:
        print(json.dumps(json.loads(packet.model_dump_json()), indent=2))
    else:
        print(format_summary(packet))
        print("  (full packet available with --json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
