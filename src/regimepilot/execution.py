"""The paper-execution boundary for the MVP (methodology approved 2026-08-27).

Two jobs, and this is the ONLY module in the project allowed to do the second:

1. ``observe_execution_state`` re-reads, immediately before an order, the
   things that must be fresh: the paper account (positions, open orders,
   options buying power), Alpaca's clock (open? minutes to close?), and a
   fresh indicative snapshot of the ONE contract Phase 4 selected. The fresh
   quote is judged by the same rules the selector used, so nothing is
   ordered against a quote the selector would have refused.

2. ``submit_paper_order`` sends one ``OrderPlan`` as a single-leg limit order,
   day, buy to open, then reads the order back once so the receipt carries
   Alpaca's status and any fill. It is the only place ``submit_order`` and an
   order request class appear anywhere in ``src/regimepilot``.

SAFETY:
* The trading client comes from ``smoke_test.build_clients``: ``paper=True``
  is hard-coded there and ``url_override`` is never passed, so this module
  cannot reach the live endpoint. Do not construct a client here.
* Every request runs through ``_guarded``: a failure becomes an
  ``ExecutionError`` (reads) or a receipt with ``submitted=False`` and only
  the exception type name (submission). Upstream text is never copied, because
  an HTTP error can quote the request it made, keys included.
* Nothing here cancels, replaces, closes or exercises anything.
* Nothing here chooses quantity or price: those arrive fixed in the plan.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from regimepilot.account import AccountError, observe_account
from regimepilot.chain import ChainError, fetch_candidate_quotes
from regimepilot.features import session_date_of, to_utc
from regimepilot.models import (
    ContractCandidate,
    ExecutionState,
    FreshQuote,
    OrderPlan,
    OrderReceipt,
    SelectedContract,
)
from regimepilot.selector import judge_candidate

__all__ = [
    "ExecutionError",
    "observe_execution_state",
    "submit_paper_order",
]


class ExecutionError(RuntimeError):
    """A pre-execution read could not be completed or understood.

    The message names the step that failed and the exception type, never the
    upstream text. A caller that sees this must not order.
    """


def _guarded(label: str, call: Callable[[], Any]) -> Any:
    """Run one step, converting any failure into a credential-safe error.

    ``from None`` drops the upstream exception instead of chaining it, so no
    traceback printed from an ExecutionError can echo an outbound request.
    """
    try:
        return call()
    except ExecutionError:
        # Already ours, and already built without upstream text.
        raise
    except Exception as error:  # noqa: BLE001 - deliberately uniform
        raise ExecutionError(f"failed to {label}: {type(error).__name__}") from None


def _as_float(value: Any) -> float | None:
    """Coerce an Alpaca scalar to float. Alpaca sends money and quantities as strings."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    """Unwrap an Alpaca enum (``OrderStatus.NEW`` -> ``"new"``)."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _as_utc(value: Any) -> datetime | None:
    return to_utc(value) if isinstance(value, datetime) else None


def _field(source: Any, name: str) -> Any:
    return getattr(source, name, None)


def observe_execution_state(
    trading_client: Any,
    option_client: Any,
    *,
    selected: SelectedContract,
    now: datetime | None = None,
) -> ExecutionState:
    """Re-read account, clock and the selected contract's quote right before ordering.

    Order of reads: account (``account.observe_account``), then the fresh
    snapshot of ``selected.symbol`` (``chain.fetch_candidate_quotes`` on the
    indicative feed), then ``get_clock`` last so the quote's age is measured
    against the server clock that stamped it. The quote's identity fields
    (type, strike, expiration, DTE) are copied from ``selected`` into a
    ``ContractCandidate`` so ``selector.judge_candidate`` can rule on it.

    Raises ``ExecutionError`` if any read fails; a partial state is never
    returned.
    """
    observed_at = to_utc(now) if now else datetime.now(timezone.utc)

    # Both upstream errors are already built without upstream text; only the
    # type changes, so a caller has one exception to catch.
    try:
        account = observe_account(trading_client, now=observed_at)
    except AccountError as error:
        raise ExecutionError(str(error)) from None

    try:
        snapshots = fetch_candidate_quotes(option_client, [selected.symbol])
    except ChainError as error:
        raise ExecutionError(str(error)) from None

    clock = _guarded("read market clock", lambda: trading_client.get_clock())
    server_time = _as_utc(_field(clock, "timestamp"))
    is_open = _field(clock, "is_open")
    next_close = _as_utc(_field(clock, "next_close"))
    minutes_to_close = (
        None
        if server_time is None or next_close is None
        else (next_close - server_time).total_seconds() / 60
    )

    # A symbol the feed was silent about is absent from the reply; it keeps
    # its identity and null quote fields, which the judge reads as no_quote.
    reference = server_time or observed_at
    quote = _field(snapshots.get(selected.symbol), "latest_quote")
    candidate = ContractCandidate(
        symbol=selected.symbol,
        option_type=selected.option_type,
        strike_price=selected.strike_price,
        expiration_date=selected.expiration_date,
        days_to_expiration=(selected.expiration_date - session_date_of(reference)).days,
        status="active",
        tradable=True,
        bid=_as_float(_field(quote, "bid_price")),
        ask=_as_float(_field(quote, "ask_price")),
        quote_at=_as_utc(_field(quote, "timestamp")),
    )
    reject_reason = judge_candidate(candidate, option_type=selected.option_type, reference=reference)

    return ExecutionState(
        observed_at=observed_at,
        account=account,
        market_is_open=None if is_open is None else bool(is_open),
        minutes_to_close=minutes_to_close,
        quote=FreshQuote(
            symbol=selected.symbol,
            bid=candidate.bid,
            ask=candidate.ask,
            quote_at=candidate.quote_at,
            server_time=server_time,
            reject_reason=reject_reason,
        ),
    )


def _limit_order_request(plan: OrderPlan) -> LimitOrderRequest:
    """The one request shape this project sends. No order class, no legs, no notional."""
    return LimitOrderRequest(
        symbol=plan.symbol,
        qty=plan.qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=plan.limit_price,
        client_order_id=plan.client_order_id,
        position_intent=PositionIntent.BUY_TO_OPEN,
    )


def _receipt(order: Any, *, order_id: str, client_order_id: str, error: str | None) -> OrderReceipt:
    return OrderReceipt(
        submitted=True,
        order_id=order_id,
        client_order_id=client_order_id,
        status=_as_text(_field(order, "status")),
        submitted_at=_as_utc(_field(order, "submitted_at")),
        filled_qty=_as_float(_field(order, "filled_qty")),
        filled_avg_price=_as_float(_field(order, "filled_avg_price")),
        error=error,
    )


def submit_paper_order(trading_client: Any, plan: OrderPlan) -> OrderReceipt:
    """Submit one plan to the PAPER account and read the order back once.

    Sends exactly: symbol, qty, side=buy, limit, limit_price, time_in_force=
    day, position_intent=buy_to_open, client_order_id. Never raises for an
    Alpaca refusal: the receipt says ``submitted=False`` with the exception
    type in ``error``. If the read-back fails after a successful submission,
    the receipt still says ``submitted=True`` with the order id and the
    read-back failure named in ``error``.
    """
    # The model already fixes these; checked again because this is the one
    # line in the project that can spend money.
    if (
        plan.side != "buy"
        or plan.order_type != "limit"
        or plan.time_in_force != "day"
        or plan.qty < 1
    ):
        raise ExecutionError(
            "refusing to submit: the plan is not a buy, limit, day order for at least one contract"
        )

    try:
        order = _guarded("submit order", lambda: trading_client.submit_order(_limit_order_request(plan)))
    except ExecutionError as error:
        return OrderReceipt(submitted=False, client_order_id=plan.client_order_id, error=str(error))

    raw_id = _field(order, "id")
    if raw_id is None:
        return OrderReceipt(
            submitted=False,
            client_order_id=plan.client_order_id,
            error="failed to submit order: reply carried no order id",
        )
    order_id = str(raw_id)

    try:
        back = _guarded("read back order", lambda: trading_client.get_order_by_id(order_id))
    except ExecutionError as error:
        return _receipt(order, order_id=order_id, client_order_id=plan.client_order_id, error=str(error))
    if back is None:
        return _receipt(
            order,
            order_id=order_id,
            client_order_id=plan.client_order_id,
            error="failed to read back order: reply was empty",
        )

    return _receipt(back, order_id=order_id, client_order_id=plan.client_order_id, error=None)
