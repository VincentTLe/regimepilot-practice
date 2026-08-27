"""Read-only paper-account state for Phase 5A.

Which positions the paper account holds, which orders are still open, and
the two balances the risk layer will need, as one immutable ``AccountState``.
The ``already_in_position`` gate reads its answer from here instead of from
the placeholder it used before this phase.

SAFETY: this module is read-only by design, exactly like every other Alpaca
boundary in this project. It calls ``get_account``, ``get_all_positions``
and ``get_orders`` and nothing else. It contains no function that submits,
cancels or replaces an order, and none that closes or exercises a position.
Do not add one here.

Because the state feeds a trading gate, its failure rule is stricter than the
observers': anything this module does not understand is an ``AccountError``,
never an empty account. A request that fails, a reply that is null, an option
position whose symbol cannot be parsed, a multi-leg order without its legs,
and an order list that may have been truncated all end the observation.
"Unknown" stays distinguishable from "confirmed empty", and a caller that
cannot tell them apart must not trade.

A SPY option is identified the way Phase 4A queries for one: an option
(asset class ``us_option``) whose OCC root symbol is exactly ``SPY``.
alpaca-py's ``Position`` and ``Order`` carry no underlying field, so the root
is parsed from the fixed-width OCC symbol; nothing is matched by prefix.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from regimepilot.config import ConfigError, load_settings
from regimepilot.console import tolerant_console
from regimepilot.features import to_utc
from regimepilot.models import UNDERLYING_SYMBOL, AccountState, OpenOrderSummary, PositionSummary

# Reused rather than reimplemented, so masking and client construction behave
# identically in every phase.
from regimepilot.smoke_test import build_clients, mask_account_id

UNDERLYING = UNDERLYING_SYMBOL

# The most open orders one request can return; Alpaca pages nothing here. A
# reply this long may be missing orders, and that is an error, not a list.
OPEN_ORDER_LIMIT = 500

# Alpaca's asset class for an option contract, the only one a SPY option can have.
OPTION_ASSET_CLASS = "us_option"

# The compact OCC symbol as Alpaca writes it: root, YYMMDD, C or P, and the
# strike times 1000 in eight digits. The tail is fixed-width, so the root is
# exactly what precedes it, and an adjusted root such as SPY1 cannot pass as SPY.
_OCC_SYMBOL = re.compile(r"([A-Z]{1,6})(\d{6})([CP])(\d{8})")

__all__ = [
    "OPEN_ORDER_LIMIT",
    "AccountError",
    "format_summary",
    "is_spy_option",
    "normalize_order",
    "normalize_position",
    "observe_account",
    "occ_root",
    "main",
]


class AccountError(RuntimeError):
    """A read-only account request could not be completed or understood.

    The message names the step that failed and the exception type, never the
    upstream text: an HTTP client's message can quote the request it made,
    which would put a key in a log.
    """


def _guarded(label: str, call: Callable[[], Any]) -> Any:
    """Run one step, converting any failure into a credential-safe error.

    ``from None`` drops the upstream exception instead of chaining it, so no
    traceback printed from an AccountError can echo an outbound request.
    """
    try:
        return call()
    except AccountError:
        # Already ours, and already built without upstream text.
        raise
    except Exception as error:  # noqa: BLE001 - deliberately uniform
        raise AccountError(f"failed to read {label}: {type(error).__name__}") from None


def _as_float(value: Any) -> float | None:
    """Coerce an Alpaca scalar to float. Alpaca sends money and quantities as strings."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    """Unwrap an Alpaca enum (``AssetClass.US_OPTION`` -> ``"us_option"``)."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _field(source: Any, name: str) -> Any:
    return getattr(source, name, None)


def occ_root(symbol: Any) -> str | None:
    """The root of a compact OCC option symbol, or ``None`` if it is not one.

    ``"SPY260902C00765000"`` -> ``"SPY"``. The symbol is taken exactly as
    given: nothing is upper-cased or stripped, because a symbol that needs
    repairing is not one this module should vouch for.
    """
    if not isinstance(symbol, str):
        return None
    match = _OCC_SYMBOL.fullmatch(symbol)
    return None if match is None else match.group(1)


def is_spy_option(symbol: Any, asset_class: Any) -> bool:
    """Whether one position or order line is a SPY option contract.

    The SDK's asset class is trusted when present: anything that is not an
    option (the SPY ETF itself is ``us_equity``) is never a SPY option. The
    underlying is then read from the OCC root, which must be exactly ``SPY``.
    A missing asset class (a multi-leg parent) is decided by the root alone.
    """
    asset_class_text = _as_text(asset_class)
    if asset_class_text is not None and asset_class_text != OPTION_ASSET_CLASS:
        return False
    return occ_root(symbol) == UNDERLYING


def normalize_position(position: Any) -> PositionSummary:
    """Keep the identity and size of one position.

    Raises ``AccountError`` for a position with no symbol, or an option whose
    symbol is not an OCC symbol: such a line cannot be classified, and an
    unclassifiable option must not read as "not SPY".
    """
    symbol = _field(position, "symbol")
    if not isinstance(symbol, str) or not symbol:
        raise AccountError("failed to read positions: a position has no symbol")

    asset_class = _as_text(_field(position, "asset_class"))
    if asset_class == OPTION_ASSET_CLASS and occ_root(symbol) is None:
        raise AccountError("failed to read positions: an option position has an unrecognized symbol")

    return PositionSummary(
        symbol=symbol,
        asset_class=asset_class,
        side=_as_text(_field(position, "side")),
        qty=_as_float(_field(position, "qty")),
        is_spy_option=is_spy_option(symbol, asset_class),
    )


def _order_is_spy_option(order: Any, legs: Sequence[Any]) -> bool:
    if is_spy_option(_field(order, "symbol"), _field(order, "asset_class")):
        return True
    return any(is_spy_option(_field(leg, "symbol"), _field(leg, "asset_class")) for leg in legs)


def normalize_order(order: Any) -> OpenOrderSummary:
    """Keep the identity of one open order.

    A multi-leg parent carries no symbol of its own and is judged by its
    legs; without them it cannot be judged, which is an ``AccountError``.
    """
    symbol = _field(order, "symbol")
    legs = list(_field(order, "legs") or [])
    if symbol is None and not legs:
        raise AccountError("failed to read open orders: an order has no symbol and no legs")

    return OpenOrderSummary(
        order_id=str(_field(order, "id") or ""),
        symbol=None if symbol is None else str(symbol),
        asset_class=_as_text(_field(order, "asset_class")),
        side=_as_text(_field(order, "side")),
        qty=_as_float(_field(order, "qty")),
        status=_as_text(_field(order, "status")),
        is_spy_option=_order_is_spy_option(order, legs),
    )


def _reply_list(label: str, reply: Any) -> list[Any]:
    """The list a reply must be. A null or non-list reply is unknown, not empty."""
    if isinstance(reply, (list, tuple)):
        return list(reply)
    raise AccountError(f"failed to read {label}: reply was not a list")


def _open_orders_request() -> GetOrdersRequest:
    # ``status`` is filtered by Alpaca, which knows which order states are
    # still open; ``nested`` rolls multi-leg legs up under their parent.
    return GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=OPEN_ORDER_LIMIT, nested=True)


def observe_account(trading_client: Any, *, now: datetime | None = None) -> AccountState:
    """Take one read-only snapshot of the paper account.

    Clients are injected so unit tests can run this without a network call.
    Raises ``AccountError`` if any request fails or any reply cannot be
    understood; a partial or guessed state is never returned.
    """
    observed_at = to_utc(now) if now else datetime.now(timezone.utc)

    account = _guarded("account", lambda: trading_client.get_account())
    if account is None:
        raise AccountError("failed to read account: reply was empty")

    positions = _guarded(
        "positions",
        lambda: tuple(
            normalize_position(position)
            for position in _reply_list("positions", trading_client.get_all_positions())
        ),
    )
    open_orders = _guarded(
        "open orders",
        lambda: tuple(
            normalize_order(order)
            for order in _reply_list(
                "open orders", trading_client.get_orders(_open_orders_request())
            )
        ),
    )
    if len(open_orders) >= OPEN_ORDER_LIMIT:
        raise AccountError(
            f"failed to read open orders: the reply filled the {OPEN_ORDER_LIMIT}-order "
            "limit, so the list may be incomplete"
        )

    return AccountState(
        observed_at=observed_at,
        account_id_masked=mask_account_id(_field(account, "id")),
        equity=_as_float(_field(account, "equity")),
        options_buying_power=_as_float(_field(account, "options_buying_power")),
        positions=positions,
        open_orders=open_orders,
        has_open_spy_option_position=any(p.is_spy_option for p in positions),
        has_open_spy_option_order=any(o.is_spy_option for o in open_orders),
    )


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def _quantity(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def _flag(value: bool) -> str:
    return "YES" if value else "no"


def format_summary(state: AccountState) -> str:
    """Balances, every position and open order, and the two SPY flags."""
    spy_positions = sum(1 for p in state.positions if p.is_spy_option)
    spy_orders = sum(1 for o in state.open_orders if o.is_spy_option)
    lines = [
        f"RegimePilot account  {state.account_id_masked}  @ "
        f"{state.observed_at.strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"  {'equity':<15} {_number(state.equity)}",
        f"  {'options bp':<15} {_number(state.options_buying_power)}",
        f"  {'positions':<15} {len(state.positions)}   ({spy_positions} SPY option)",
    ]
    for position in state.positions:
        lines.append(
            f"    {position.symbol:<22} {position.asset_class or '-':<10} "
            f"{position.side or '-':<6} qty {_quantity(position.qty):>6}"
            + ("   SPY option" if position.is_spy_option else "")
        )
    lines.append(f"  {'open orders':<15} {len(state.open_orders)}   ({spy_orders} SPY option)")
    for order in state.open_orders:
        lines.append(
            f"    {order.symbol or '(multi-leg)':<22} {order.asset_class or '-':<10} "
            f"{order.side or '-':<6} qty {_quantity(order.qty):>6}   {order.status or '-'}"
            + ("   SPY option" if order.is_spy_option else "")
        )
    lines.append(
        f"  {'spy option':<15} position {_flag(state.has_open_spy_option_position)}"
        f"   open order {_flag(state.has_open_spy_option_order)}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the paper account state, or the state itself with ``--json``. Reads only."""
    tolerant_console()
    arguments = list(sys.argv[1:] if argv is None else argv)

    try:
        settings = load_settings()
        trading_client, _data_client = build_clients(settings)
    except ConfigError as error:
        # ConfigError messages are built by us and never contain a credential.
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    try:
        state = observe_account(trading_client)
    except AccountError as error:
        print(f"account read failed: {error}", file=sys.stderr)
        return 1

    if "--json" in arguments:
        print(json.dumps(json.loads(state.model_dump_json()), indent=2))
    else:
        print(format_summary(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
