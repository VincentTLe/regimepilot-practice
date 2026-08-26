"""Read-only option-chain observation for Phase 4A.

Which SPY contracts are on offer around the money for one proposal direction,
and what the indicative feed quotes for them.

SAFETY: this module is read-only by design, exactly like Phase 2A's observer
and Phase 2B's history. It contains no function that submits, cancels or
replaces an order, and none that closes or exercises a position. Do not add
one here.

This phase observes and does not judge. A ChainPacket lists every contract in
the query window with the quote it had, and carries no verdict, no ranking,
no threshold and no greek. Deciding which contract (if any) is acceptable is
Phase 4B's job, and the numbers it will need are chosen from what this module
prints during market hours.

Two failure modes are kept strictly apart, as in every Alpaca boundary here:

* A call succeeds but carries no data -> a null field, or no candidates. A
  contract the feed is silent about keeps its identity and null quote fields.
* A call fails -> ``ChainError``. A partial packet is never returned and a
  missing value is never invented.

Every option request names ``OptionsFeed.INDICATIVE`` explicitly. Alpaca's
default is "the best feed your subscription allows", which would silently
change the meaning of every quote the day a subscription changes.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

from alpaca.data.enums import OptionsFeed
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from regimepilot.config import ConfigError, Settings, load_settings
from regimepilot.features import session_date_of, spread_bps, to_utc
from regimepilot.history import HistoryError, fetch_latest_quote
from regimepilot.models import UNDERLYING_SYMBOL, ChainPacket, ContractCandidate, TradeAction
from regimepilot.smoke_test import build_clients

UNDERLYING = UNDERLYING_SYMBOL

# Never left to the account default: this phase is defined on the free
# indicative feed, which is the feed the account actually has.
OPTION_FEED = OptionsFeed.INDICATIVE

# Approved for Phase 4 on 2026-08-25: 5-10 calendar days, counted from the
# New York date. Deliberately narrower than Phase 2A's 3-14 day universe,
# which observed what exists; this window is where a contract may be chosen.
MIN_DAYS_TO_EXPIRATION = 5
MAX_DAYS_TO_EXPIRATION = 10

# Dollars either side of the SPY midpoint to query. A temporary bound that
# keeps the request small (SPY has $1 strikes near the money). It is NOT a
# selection or fallback rule: whether a strike this far from the money is
# acceptable is a Phase 4B decision that has not been made.
QUERY_STRIKE_WINDOW = 5.0

# A 5-10 day window of $1 strikes is well under one page; the cap only exists
# so an unexpected reply cannot page forever. Running past it fails the
# observation rather than truncating it, as in Phase 2A.
OPTION_CONTRACT_PAGE_LIMIT = 1000
MAX_OPTION_CONTRACT_PAGES = 5

# The snapshot endpoint accepts at most 100 contract symbols per request.
SNAPSHOT_SYMBOLS_PER_REQUEST = 100

# The only two directions that name a contract type. HOLD looks at nothing.
_CONTRACT_TYPE_FOR_ACTION: dict[str, ContractType] = {
    "BUY_CALL": ContractType.CALL,
    "BUY_PUT": ContractType.PUT,
}

__all__ = [
    "ChainError",
    "build_option_data_client",
    "fetch_candidate_contracts",
    "fetch_candidate_quotes",
    "fetch_underlying_mid",
    "format_summary",
    "normalize_candidate",
    "observe_chain",
    "main",
]


class ChainError(RuntimeError):
    """A read-only option-data request could not be completed.

    The message names the step that failed and the exception type, never the
    upstream text: an HTTP client's message can quote the request it made,
    which would put a key in a log.
    """


def build_option_data_client(settings: Settings) -> OptionHistoricalDataClient:
    """Create the read-only option market-data client from paper credentials.

    Market data has no paper/live switch of its own, but the same refusal as
    ``build_clients`` keeps every client in this project behind one guard.
    """
    if not settings.paper:
        raise ConfigError("Refusing to build option data client: paper trading is not enabled.")

    return OptionHistoricalDataClient(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
    )


def _guarded(label: str, call: Callable[[], Any]) -> Any:
    """Run one request, converting any failure into a credential-safe error.

    ``from None`` drops the upstream exception instead of chaining it, so no
    traceback printed from a ChainError can echo an outbound request.
    """
    try:
        return call()
    except ChainError:
        # Already ours, and already built without upstream text.
        raise
    except Exception as error:  # noqa: BLE001 - deliberately uniform
        raise ChainError(f"failed to read {label}: {type(error).__name__}") from None


def _as_float(value: Any) -> float | None:
    """Coerce an Alpaca scalar to float. Alpaca sends strikes as strings."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    """Unwrap an Alpaca enum (``ContractType.CALL`` -> ``"call"``)."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _field(source: Any, name: str) -> Any:
    return getattr(source, name, None)


def fetch_underlying_mid(data_client: Any) -> float | None:
    """SPY midpoint from the top of book, or ``None`` if the quote is unusable.

    Reuses Phase 2B's quote read (IEX, credential-safe) and its validity rule:
    a midpoint exists only when ``spread_bps`` accepts the bid/ask pair.
    """
    try:
        bid, ask = fetch_latest_quote(data_client, symbol=UNDERLYING)
    except HistoryError as error:
        # Already credential-safe; only the type changes, so a caller of this
        # module has one exception to catch.
        raise ChainError(str(error)) from None

    if spread_bps(bid, ask) is None:
        return None
    return (bid + ask) / 2


def fetch_candidate_contracts(
    trading_client: Any,
    *,
    contract_type: ContractType,
    underlying_mid: float,
    today: date,
) -> list[Any]:
    """Page the contract endpoint for one type, the DTE window and the strike band.

    Every filter is applied by Alpaca, so the reply is already the narrow
    slice this phase looks at. Raises ``ChainError`` if the window needs more
    pages than the cap allows, because a complete slice is the only kind this
    observer reports.
    """
    contracts: list[Any] = []
    page_token: str | None = None

    for _ in range(MAX_OPTION_CONTRACT_PAGES):
        request = GetOptionContractsRequest(
            underlying_symbols=[UNDERLYING],
            root_symbol=UNDERLYING,
            status=AssetStatus.ACTIVE,
            type=contract_type,
            expiration_date_gte=today + timedelta(days=MIN_DAYS_TO_EXPIRATION),
            expiration_date_lte=today + timedelta(days=MAX_DAYS_TO_EXPIRATION),
            # alpaca-py types the strike bounds as strings; a float is rejected.
            strike_price_gte=f"{underlying_mid - QUERY_STRIKE_WINDOW:.2f}",
            strike_price_lte=f"{underlying_mid + QUERY_STRIKE_WINDOW:.2f}",
            limit=OPTION_CONTRACT_PAGE_LIMIT,
            page_token=page_token,
        )
        response = _guarded(
            "option contracts", lambda: trading_client.get_option_contracts(request)
        )
        contracts.extend(_field(response, "option_contracts") or [])
        page_token = _field(response, "next_page_token")
        if not page_token:
            return contracts

    raise ChainError(
        "failed to read option contracts: more pages remain after the "
        f"{MAX_OPTION_CONTRACT_PAGES}-page limit, so the window is incomplete"
    )


def _snapshot_rows(response: Any) -> dict[str, Any]:
    """The ``{symbol: OptionsSnapshot}`` dict ``get_option_snapshot`` returns.

    Unlike bars (``.data[symbol]``) and news (``.data["news"]``), the typed
    option snapshot reply is a plain dict keyed by contract symbol. A symbol
    the feed returned as null is simply absent from it.
    """
    data = response if isinstance(response, dict) else _field(response, "data")
    return dict(data) if isinstance(data, dict) else {}


def fetch_candidate_quotes(option_client: Any, symbols: Sequence[str]) -> dict[str, Any]:
    """Latest indicative snapshot per symbol, in batches the endpoint accepts."""
    snapshots: dict[str, Any] = {}
    ordered = list(symbols)

    for start in range(0, len(ordered), SNAPSHOT_SYMBOLS_PER_REQUEST):
        batch = ordered[start : start + SNAPSHOT_SYMBOLS_PER_REQUEST]
        request = OptionSnapshotRequest(symbol_or_symbols=batch, feed=OPTION_FEED)
        response = _guarded(
            "option snapshots", lambda: option_client.get_option_snapshot(request)
        )
        snapshots.update(_snapshot_rows(response))

    return snapshots


def _fetch_clock_time(trading_client: Any) -> datetime | None:
    """Alpaca's own idea of now, from the market clock; ``None`` if it has none.

    Read after the quotes so a quote's age can be measured against the clock
    that stamped it. The observing machine's clock was found fourteen seconds
    slow, which would make every quote look fourteen seconds fresher than it
    is; a freshness threshold chosen from that would be wrong by the same
    amount.
    """
    clock = _guarded("market clock", trading_client.get_clock)
    stamp = _field(clock, "timestamp")
    return to_utc(stamp) if isinstance(stamp, datetime) else None


def normalize_candidate(contract: Any, snapshot: Any, *, today: date) -> ContractCandidate:
    """Join one contract's identity with its snapshot. No snapshot in, null quote out."""
    expiration = _as_date(_field(contract, "expiration_date"))
    quote = _field(snapshot, "latest_quote")
    quote_at = _field(quote, "timestamp")
    tradable = _field(contract, "tradable")

    return ContractCandidate(
        symbol=str(_field(contract, "symbol") or ""),
        option_type=_as_text(_field(contract, "type")),
        strike_price=_as_float(_field(contract, "strike_price")),
        expiration_date=expiration,
        days_to_expiration=None if expiration is None else (expiration - today).days,
        status=_as_text(_field(contract, "status")),
        tradable=None if tradable is None else bool(tradable),
        bid=_as_float(_field(quote, "bid_price")),
        ask=_as_float(_field(quote, "ask_price")),
        quote_at=to_utc(quote_at) if isinstance(quote_at, datetime) else None,
    )


def _sort_key(candidate: ContractCandidate) -> tuple[date, float, str]:
    strike = float("inf") if candidate.strike_price is None else candidate.strike_price
    return (candidate.expiration_date or date.max, strike, candidate.symbol)


def observe_chain(
    trading_client: Any,
    data_client: Any,
    option_client: Any,
    *,
    action: TradeAction,
    now: datetime | None = None,
) -> ChainPacket:
    """Take one read-only look at the chain slice a proposal direction draws from.

    ``HOLD`` observes nothing and makes no request: there is no direction to
    look along. A missing or unusable SPY quote yields no candidates, because
    there is no midpoint to build a strike window around. Raises
    ``ChainError`` if any request fails; a partial packet is never returned.
    """
    observed_at = to_utc(now) if now else datetime.now(timezone.utc)
    feed = OPTION_FEED.value

    contract_type = _CONTRACT_TYPE_FOR_ACTION.get(action)
    if contract_type is None:
        return ChainPacket(observed_at=observed_at, action=action, option_feed=feed)

    underlying_mid = fetch_underlying_mid(data_client)
    if underlying_mid is None:
        return ChainPacket(observed_at=observed_at, action=action, option_feed=feed)

    today = session_date_of(observed_at)
    contracts = fetch_candidate_contracts(
        trading_client, contract_type=contract_type, underlying_mid=underlying_mid, today=today
    )
    symbols = [str(_field(contract, "symbol") or "") for contract in contracts]
    snapshots = fetch_candidate_quotes(option_client, [s for s in symbols if s]) if symbols else {}
    quotes_read_at = _fetch_clock_time(trading_client)

    candidates = sorted(
        (
            normalize_candidate(contract, snapshots.get(symbol), today=today)
            for contract, symbol in zip(contracts, symbols)
        ),
        key=_sort_key,
    )
    return ChainPacket(
        observed_at=observed_at,
        action=action,
        option_feed=feed,
        underlying_mid=underlying_mid,
        quotes_read_at=quotes_read_at,
        candidates=tuple(candidates),
    )


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _signed(value: float | None) -> str:
    return "-" if value is None else f"{value:+.2f}"


def _stamp(value: datetime | None) -> str:
    return "-" if value is None else value.strftime("%Y-%m-%d %H:%M:%SZ")


def _clock(value: datetime | None) -> str:
    return "-" if value is None else value.strftime("%H:%M:%SZ")


def quote_age_seconds(candidate: ContractCandidate, reference: datetime) -> float | None:
    """How old the candidate's quote was at ``reference``; never negative.

    Pass ``packet.quotes_read_at`` (Alpaca's clock) when it exists; the
    packet's ``observed_at`` is the local clock and only a fallback.
    """
    if candidate.quote_at is None:
        return None
    return max(0.0, (to_utc(reference) - candidate.quote_at).total_seconds())


def format_summary(packet: ChainPacket) -> str:
    """One line per candidate: identity, quote, spread and quote age. No verdicts."""
    mid = packet.underlying_mid
    lines = [
        f"RegimePilot chain  {packet.symbol}  {packet.action}  @ {_stamp(packet.observed_at)}",
        f"  {'feed':<15} {packet.option_feed}",
        f"  {'underlying mid':<15} {_number(mid)}",
    ]

    if packet.action not in _CONTRACT_TYPE_FOR_ACTION:
        lines.append("  (HOLD: no direction, so no chain was observed)")
        return "\n".join(lines)
    if mid is None:
        lines.append("  (no usable SPY quote, so no strike window was queried)")
        return "\n".join(lines)

    if packet.quotes_read_at is None:
        reference = packet.observed_at
        lines.append(f"  {'alpaca clock':<15} -   (ages below use observed_at, the local clock)")
    else:
        reference = packet.quotes_read_at
        skew = (packet.quotes_read_at - packet.observed_at).total_seconds()
        lines.append(
            f"  {'alpaca clock':<15} {_stamp(packet.quotes_read_at)}"
            f"   ({skew:+.1f} s vs observed_at; ages below use the Alpaca clock)"
        )
    lines.append(
        f"  {'query window':<15} {MIN_DAYS_TO_EXPIRATION}-{MAX_DAYS_TO_EXPIRATION} DTE"
        f"   strike {_number(mid - QUERY_STRIKE_WINDOW)} -> {_number(mid + QUERY_STRIKE_WINDOW)}"
    )
    lines.append(f"  {'candidates':<15} {len(packet.candidates)}")
    if not packet.candidates:
        lines.append("  (no contracts in the query window)")
        return "\n".join(lines)

    lines.append(
        f"  {'expiration':<11}{'dte':>4}{'strike':>9}{'k-mid':>8}{'bid':>8}{'ask':>8}"
        f"{'spread':>8}{'bps':>8}  {'quote at':<10}{'age s':>8}  tradable"
    )
    for candidate in packet.candidates:
        bid, ask = candidate.bid, candidate.ask
        spread = None if bid is None or ask is None else ask - bid
        strike = candidate.strike_price
        moneyness = None if strike is None else strike - mid
        dte = "-" if candidate.days_to_expiration is None else str(candidate.days_to_expiration)
        tradable = "-" if candidate.tradable is None else ("yes" if candidate.tradable else "NO")
        lines.append(
            f"  {str(candidate.expiration_date or '-'):<11}{dte:>4}{_number(strike):>9}"
            f"{_signed(moneyness):>8}{_number(bid):>8}{_number(ask):>8}{_number(spread):>8}"
            f"{_number(spread_bps(bid, ask), 1):>8}  {_clock(candidate.quote_at):<10}"
            f"{_number(quote_age_seconds(candidate, reference), 1):>8}  {tradable}"
        )
    return "\n".join(lines)


def _action_argument(arguments: Sequence[str]) -> str | None:
    """The value of ``--action X`` or ``--action=X``, upper-cased, or ``None``."""
    for index, argument in enumerate(arguments):
        if argument == "--action" and index + 1 < len(arguments):
            return arguments[index + 1].strip().upper()
        if argument.startswith("--action="):
            return argument.split("=", 1)[1].strip().upper()
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Print the chain slice for one direction, or the packet itself with ``--json``.

    ``--action`` is required and must be BUY_CALL or BUY_PUT: this command
    exists to look at real quotes for one direction without running the LLM,
    and HOLD would have nothing to show.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)

    action = _action_argument(arguments)
    if action not in _CONTRACT_TYPE_FOR_ACTION:
        print(
            "usage: python -m regimepilot.chain --action BUY_CALL|BUY_PUT [--json]",
            file=sys.stderr,
        )
        return 1

    try:
        settings = load_settings()
        trading_client, data_client = build_clients(settings)
        option_client = build_option_data_client(settings)
    except ConfigError as error:
        # ConfigError messages are built by us and never contain a credential.
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    try:
        packet = observe_chain(
            trading_client, data_client, option_client, action=action  # type: ignore[arg-type]
        )
    except ChainError as error:
        print(f"chain read failed: {error}", file=sys.stderr)
        return 1

    if "--json" in arguments:
        print(json.dumps(json.loads(packet.model_dump_json()), indent=2))
    else:
        print(format_summary(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
