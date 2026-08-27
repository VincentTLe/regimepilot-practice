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

from datetime import datetime
from typing import Any

from regimepilot.models import ExecutionState, OrderPlan, OrderReceipt, SelectedContract

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
    raise NotImplementedError("Worker B implements this")


def submit_paper_order(trading_client: Any, plan: OrderPlan) -> OrderReceipt:
    """Submit one plan to the PAPER account and read the order back once.

    Sends exactly: symbol, qty, side=buy, limit, limit_price, time_in_force=
    day, position_intent=buy_to_open, client_order_id. Never raises for an
    Alpaca refusal: the receipt says ``submitted=False`` with the exception
    type in ``error``. If the read-back fails after a successful submission,
    the receipt still says ``submitted=True`` with the order id and the
    read-back failure named in ``error``.
    """
    raise NotImplementedError("Worker B implements this")
