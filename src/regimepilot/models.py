"""Normalized, network-free models for one market observation.

Nothing here imports Alpaca. An ``ObservationPacket`` is entirely our own data,
so an SDK response object can never travel further than the observer that
converted it.

Every model is frozen: an observation is a record of what was true at
``observed_at``, not a scratch pad for a later phase to write into.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

UNDERLYING_SYMBOL = "SPY"


def _to_utc(value: datetime) -> datetime:
    """Normalize a timestamp to UTC. A naive value is assumed to already be UTC.

    Alpaca reports the clock in market-local time and market data in UTC;
    normalizing here means a consumer never has to ask which one it is holding.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_to_utc)]


class Observation(BaseModel):
    """Base for every observation model: immutable, and closed to stray fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class OhlcvBar(Observation):
    """One OHLCV bar. Any field may be null if the feed omitted it."""

    timestamp: UtcDatetime | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None


class MarketState(Observation):
    """Whether the equity market is open, and when it next changes."""

    is_open: bool | None = None
    next_open: UtcDatetime | None = None
    next_close: UtcDatetime | None = None


class AccountSnapshot(Observation):
    """Paper account balances.

    ``account_id_masked`` is the only identity field that exists. The real
    account id is never stored on this model, so it cannot be serialized.
    """

    account_id_masked: str
    equity: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    options_buying_power: float | None = None
    options_trading_level: int | None = None


class UnderlyingSnapshot(Observation):
    """Last trade, top of book and recent bars for the underlying."""

    symbol: str = UNDERLYING_SYMBOL
    latest_trade_price: float | None = None
    latest_trade_timestamp: UtcDatetime | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    quote_timestamp: UtcDatetime | None = None
    minute_bar: OhlcvBar | None = None
    daily_bar: OhlcvBar | None = None
    previous_daily_bar: OhlcvBar | None = None


class OptionContractSummary(Observation):
    """The identity of one option contract, and nothing else.

    Deliberately carries no price, no greek and no ranking: Phase 2A observes
    which contracts exist, it does not judge them.
    """

    symbol: str
    option_type: str | None = None
    strike_price: float | None = None
    expiration_date: date | None = None
    status: str | None = None
    tradable: bool | None = None


class OptionUniverse(Observation):
    """Every option contract this observation looked at.

    ``contract_count`` counts ``contracts``; the observer pages the endpoint to
    exhaustion so the count and the expiration bounds describe the whole
    requested window rather than one page of it.

    ``contracts`` is a tuple, not a list. Freezing the model alone would still
    have left the collection open: a holder could append a contract that was
    never observed, or drop one that was, while ``contract_count`` and the
    expiration bounds went on describing the original observation.
    """

    contract_count: int = 0
    earliest_expiration: date | None = None
    latest_expiration: date | None = None
    contracts: tuple[OptionContractSummary, ...] = ()


class ObservationPacket(Observation):
    """One complete, self-consistent read-only view of the market."""

    observed_at: UtcDatetime
    market: MarketState
    account: AccountSnapshot
    underlying: UnderlyingSnapshot
    option_universe: OptionUniverse


class NewsItem(Observation):
    """One filtered headline suitable for LLM briefing."""

    id: int
    headline: str
    summary: str
    age_minutes: float
    symbols: tuple[str, ...] = ()
    source: str | None = None


class NewsPacket(Observation):
    """Recent headlines for the underlying, capped and relevance-filtered."""

    observed_at: UtcDatetime
    available: bool = False
    item_count: int = 0
    items: tuple[NewsItem, ...] = ()


class UnderlyingEvidence(Observation):
    """Underlying feature context copied into an LLM briefing."""

    market_is_open: bool | None = None
    data_feed: str
    minutes_since_open: float | None = None
    minutes_to_close: float | None = None
    spread_bps: float | None = None
    bar_age_seconds: float | None = None
    return_15m: float | None = None
    return_60m: float | None = None
    return_since_open: float | None = None
    overnight_gap_pct: float | None = None
    realized_vol_30m: float | None = None


class GatesEvidence(Observation):
    """Pre-gate outcome attached to one evidence packet."""

    passed: bool
    hold_reason: str | None = None
    momentum_align: str = "unknown"


class NewsEvidence(Observation):
    """News slice of an evidence packet."""

    available: bool = False
    item_count: int = 0
    items: tuple[NewsItem, ...] = ()


class AccountHint(Observation):
    """Kept for compatibility: true when the paper account holds any SPY option.

    The portfolio agent reads ``EvidencePacket.portfolio`` instead; this flag
    no longer gates anything.
    """

    has_open_option_position: bool = False


# ---------------------------------------------------------------------------
# Portfolio context (approved 2026-08-27): what the LLM sees about every held
# SPY option, every pending order, and whether a new entry is even eligible.
# ---------------------------------------------------------------------------


class PositionMemo(Observation):
    """What the journal remembers about one held contract: why it was opened,
    when, and what the agent last decided about it. All best effort."""

    symbol: str
    entered_at: UtcDatetime | None = None
    entry_thesis: str | None = None
    previous_decision: str | None = None


class OpenPositionContext(Observation):
    """One held SPY option as the portfolio manager sees it.

    Identity comes from the OCC symbol, size and marks from Alpaca's position
    record, memory from the journal. ``pending_order_side`` is the side of an
    open order on this exact symbol, if any: it blocks actions on this symbol
    only, never the rest of the portfolio.
    """

    symbol: str
    option_type: str
    strike_price: float
    expiration_date: date
    days_to_expiration: int
    qty: float
    avg_entry_price: float | None = None
    cost_basis: float | None = None
    current_price: float | None = None
    unrealized_pl: float | None = None
    unrealized_plpc: float | None = None
    pending_order_side: str | None = None
    entered_at: UtcDatetime | None = None
    hours_held: float | None = None
    entry_thesis: str | None = None
    previous_decision: str | None = None


class PendingOrder(Observation):
    """One open SPY option order, by exact symbol."""

    order_id: str
    symbol: str
    side: str | None = None
    qty: float | None = None
    status: str | None = None


class EntryCandidate(Observation):
    """One deterministically shortlisted contract the LLM may pick by id.

    Greeks and IV are copied when the feed provides them and null otherwise;
    a null never blocks anything. Empty shortlists mean "the deterministic
    selector will pick", which is always a valid fallback.
    """

    candidate_id: str
    symbol: str
    option_type: str
    strike_price: float
    expiration_date: date
    days_to_expiration: int
    bid: float
    ask: float
    mid: float
    spread_bps: float
    moneyness: float
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None


class PortfolioLimits(Observation):
    """The deterministic limits the LLM works under, stated so it can see them."""

    max_open_positions: int
    max_new_entries_per_cycle: int
    max_entry_premium_usd: float
    max_total_premium_usd: float


class PortfolioContext(Observation):
    """Every held SPY option, every pending order, and the entry pre-check.

    ``entry_allowed`` is a deterministic pre-check (entry gates, position
    count, total premium, pending buys); the risk layer re-checks everything
    on a fresh account read before any order. ``positions`` are ordered by
    symbol so two cycles describe the same portfolio the same way.
    """

    positions: tuple[OpenPositionContext, ...] = ()
    open_position_count: int = 0
    total_cost_basis: float | None = None
    options_buying_power: float | None = None
    equity: float | None = None
    pending_orders: tuple[PendingOrder, ...] = ()
    entry_allowed: bool = False
    entry_blocked_reason: str | None = None
    underlying_mid: float | None = None
    call_candidates: tuple[EntryCandidate, ...] = ()
    put_candidates: tuple[EntryCandidate, ...] = ()
    limits: PortfolioLimits

    @model_validator(mode="after")
    def _count_and_flag_agree(self) -> PortfolioContext:
        if self.open_position_count != len(self.positions):
            raise ValueError("open_position_count does not agree with the positions listed")
        if self.entry_allowed and self.entry_blocked_reason is not None:
            raise ValueError("an allowed entry cannot carry a blocked reason")
        if not self.entry_allowed and self.entry_blocked_reason is None:
            raise ValueError("a blocked entry must say why")
        return self


class EvidencePacket(Observation):
    """One normalized briefing for LLM portfolio reasoning.

    ``portfolio`` is what the agent manages; ``account`` is the Phase 3 flag
    kept for older readers.
    """

    observed_at: UtcDatetime
    symbol: str = UNDERLYING_SYMBOL
    gates: GatesEvidence
    underlying: UnderlyingEvidence
    news: NewsEvidence
    account: AccountHint
    portfolio: PortfolioContext | None = None


TradeAction = Literal["BUY_CALL", "BUY_PUT", "HOLD"]
Confidence = Literal["low", "medium", "high"]


class TradeProposal(Observation):
    """Direction proposal from pre-gates or LLM reasoning. No execution details."""

    observed_at: UtcDatetime
    symbol: str = UNDERLYING_SYMBOL
    action: TradeAction
    confidence: Confidence
    thesis: str
    evidence_used: tuple[str, ...] = ()
    gate_skipped: bool = False
    model: str | None = None


class ContractCandidate(Observation):
    """One option contract near the money, with the quote it had when observed.

    Identity comes from the Trading API contract list, the quote from the
    option snapshot feed. Any field but ``symbol`` may be null if the feed had
    nothing to say. Deliberately carries no greek, no implied volatility, no
    open interest and no verdict: Phase 4A observes what is on offer, it does
    not judge it.
    """

    symbol: str
    option_type: str | None = None
    strike_price: float | None = None
    expiration_date: date | None = None
    days_to_expiration: int | None = None
    status: str | None = None
    tradable: bool | None = None
    bid: float | None = None
    ask: float | None = None
    quote_at: UtcDatetime | None = None


class ChainPacket(Observation):
    """The narrow slice of the option chain one proposal direction looks at.

    ``underlying_mid`` is the SPY midpoint the strike window was built around.
    ``None`` means there was no usable SPY quote, and then there are no
    candidates because there was no window to query. ``candidates`` is a
    tuple for the same reason ``OptionUniverse.contracts`` is.

    ``observed_at`` is the observing machine's clock; ``quotes_read_at`` is
    Alpaca's own clock, read right after the quotes arrived. Quote ages are
    measured against the latter, because the machine's clock was found to be
    fourteen seconds slow and a freshness threshold must not inherit that.
    """

    observed_at: UtcDatetime
    symbol: str = UNDERLYING_SYMBOL
    action: TradeAction
    option_feed: str
    underlying_mid: float | None = None
    quotes_read_at: UtcDatetime | None = None
    candidates: tuple[ContractCandidate, ...] = ()


RejectReason = Literal[
    "invalid_contract",
    "not_tradable",
    "no_quote",
    "invalid_quote",
    "stale_quote",
    "wide_spread",
]
SelectionStatus = Literal["selected", "no_contract", "not_applicable"]
NoContractReason = Literal["no_underlying_price", "no_candidates", "all_candidates_rejected"]


class CandidateVerdict(Observation):
    """Why one candidate at the target expiration was, or was not, eligible."""

    symbol: str
    expiration_date: date | None = None
    days_to_expiration: int | None = None
    strike_price: float | None = None
    reject_reason: RejectReason | None = None


class SelectedContract(Observation):
    """The one contract Phase 4 chose, with the quote it was chosen on.

    Carries no quantity, no limit price and no order type: those are Phase 5
    decisions. The quote fields are evidence of what was true at selection,
    not an instruction.
    """

    symbol: str
    option_type: str
    strike_price: float
    expiration_date: date
    days_to_expiration: int
    bid: float
    ask: float
    mid: float
    spread_bps: float
    quote_at: UtcDatetime
    quote_age_seconds: float
    underlying_mid: float


class SelectionResult(Observation):
    """One answer to "which contract?", with the verdict on every candidate.

    ``status`` is the headline and the other fields must agree with it:
    ``selected`` carries a contract and no reason, ``no_contract`` carries a
    reason and no contract, ``not_applicable`` (a HOLD) carries neither.
    """

    observed_at: UtcDatetime
    symbol: str = UNDERLYING_SYMBOL
    action: TradeAction
    status: SelectionStatus
    reason: NoContractReason | None = None
    target_expiration: date | None = None
    selected: SelectedContract | None = None
    verdicts: tuple[CandidateVerdict, ...] = ()

    @model_validator(mode="after")
    def _status_agrees_with_its_fields(self) -> SelectionResult:
        has_contract = self.selected is not None
        has_reason = self.reason is not None
        expected = {
            "selected": (True, False),
            "no_contract": (False, True),
            "not_applicable": (False, False),
        }[self.status]
        if (has_contract, has_reason) != expected:
            raise ValueError(
                f"status {self.status!r} does not agree with "
                f"selected={'set' if has_contract else 'none'}, "
                f"reason={'set' if has_reason else 'none'}"
            )
        return self


class PositionSummary(Observation):
    """One open position: identity and size only, no price and no P&L.

    ``is_spy_option`` records the classification the account flag was built
    from, so a reader can see which line made
    ``AccountState.has_open_spy_option_position`` true.
    """

    symbol: str
    asset_class: str | None = None
    side: str | None = None
    qty: float | None = None
    is_spy_option: bool = False
    # Management facts Alpaca reports per position (money as text upstream,
    # floats here, null when absent). Read, never computed.
    avg_entry_price: float | None = None
    cost_basis: float | None = None
    current_price: float | None = None
    market_value: float | None = None
    unrealized_pl: float | None = None
    unrealized_plpc: float | None = None
    qty_available: float | None = None


class OpenOrderSummary(Observation):
    """One order Alpaca reports as open. Identity only: no price, no plan.

    A multi-leg parent order has no symbol of its own; ``is_spy_option`` is
    then true when any of its legs is a SPY option.
    """

    order_id: str
    symbol: str | None = None
    asset_class: str | None = None
    side: str | None = None
    qty: float | None = None
    status: str | None = None
    is_spy_option: bool = False


class AccountState(Observation):
    """The paper account as it was at ``observed_at``: balances, holdings, open orders.

    Exists only when every read succeeded: a failed or misunderstood read is
    an error in the reader, never an empty state here, so "unknown" can never
    be mistaken for "confirmed empty". Positions and open orders are kept
    apart, each with its own SPY-option flag, and each flag must agree with
    the lines it summarizes. Carries no quantity to trade, no premium limit
    and no order plan: those are Phase 5B decisions.
    """

    observed_at: UtcDatetime
    account_id_masked: str
    equity: float | None = None
    options_buying_power: float | None = None
    positions: tuple[PositionSummary, ...] = ()
    open_orders: tuple[OpenOrderSummary, ...] = ()
    has_open_spy_option_position: bool = False
    has_open_spy_option_order: bool = False

    @model_validator(mode="after")
    def _flags_agree_with_their_lines(self) -> AccountState:
        expected_position = any(p.is_spy_option for p in self.positions)
        expected_order = any(o.is_spy_option for o in self.open_orders)
        if self.has_open_spy_option_position != expected_position:
            raise ValueError(
                f"has_open_spy_option_position={self.has_open_spy_option_position} "
                "does not agree with the positions listed"
            )
        if self.has_open_spy_option_order != expected_order:
            raise ValueError(
                f"has_open_spy_option_order={self.has_open_spy_option_order} "
                "does not agree with the open orders listed"
            )
        return self


# ---------------------------------------------------------------------------
# Portfolio agent execution models (methodology approved 2026-08-27, with the
# three corrections of the same day). The LLM produces a PortfolioDecision;
# deterministic code turns each requested action into at most one order.
# ---------------------------------------------------------------------------


PositionAction = Literal["HOLD", "CLOSE"]
EntryDirection = Literal["CALL", "PUT"]


class PositionDecision(Observation):
    """The LLM's verdict on one held contract, by exact symbol."""

    symbol: str
    action: PositionAction
    reason: str


class EntryDecision(Observation):
    """The LLM's request for at most one new position.

    ``candidate_id`` names a shortlisted contract when the LLM chose one and
    the id was valid; ``None`` means the deterministic selector picks.
    Quantity and price never appear here.
    """

    direction: EntryDirection
    candidate_id: str | None = None
    thesis: str


class PortfolioDecision(Observation):
    """One cycle's decision over the whole portfolio.

    Every held symbol gets exactly one ``PositionDecision`` (the parser fills
    a HOLD for any the model skipped). ``new_entry`` is present only when the
    entry pre-check allowed one. ``gate_skipped`` means no model was called
    and this is the safe default (HOLD all, no entry).
    """

    observed_at: UtcDatetime
    symbol: str = UNDERLYING_SYMBOL
    positions: tuple[PositionDecision, ...] = ()
    new_entry: EntryDecision | None = None
    confidence: Confidence
    portfolio_thesis: str
    evidence_used: tuple[str, ...] = ()
    gate_skipped: bool = False
    model: str | None = None

    @model_validator(mode="after")
    def _one_decision_per_symbol(self) -> PortfolioDecision:
        symbols = [p.symbol for p in self.positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("a position decision names the same symbol twice")
        return self


class FreshQuote(Observation):
    """The contract re-quoted immediately before an order.

    ``server_time`` is Alpaca's clock read right after the quote, so the age
    is measured the way Phase 4 measured it. ``reject_reason`` is the verdict
    of the same rules ``selector.judge_candidate`` applies (identity taken
    from the contract, bid/ask/timestamp from the fresh snapshot); ``None``
    means the quote is acceptable to order against.
    """

    symbol: str
    bid: float | None = None
    ask: float | None = None
    quote_at: UtcDatetime | None = None
    server_time: UtcDatetime | None = None
    reject_reason: RejectReason | None = None


class ExecutionState(Observation):
    """Everything re-read right before an order: account, clock, fresh quote.

    Exists only when every read succeeded; a failed read is an error in the
    reader, never a partial state here. ``minutes_to_close`` is measured on
    Alpaca's clock (``next_close - timestamp``), not the local one.
    """

    observed_at: UtcDatetime
    account: AccountState
    market_is_open: bool | None = None
    minutes_to_close: float | None = None
    quote: FreshQuote


OrderSideLiteral = Literal["buy", "sell"]
PositionIntentLiteral = Literal["buy_to_open", "sell_to_close"]


class OrderPlan(Observation):
    """One fully deterministic single-leg option order. Nothing here comes from the LLM.

    Two shapes exist: buy to open (entry, limit at the fresh ask) and sell to
    close (exit, limit at the fresh bid). Both are limit, day. ``qty`` and
    ``limit_price`` are set by ``risk`` alone. ``notional_usd`` is
    ``limit_price * 100 * qty``: the most an entry can cost, the least an
    exit brings in.
    """

    symbol: str
    side: OrderSideLiteral = "buy"
    qty: int
    order_type: Literal["limit"] = "limit"
    time_in_force: Literal["day"] = "day"
    position_intent: PositionIntentLiteral = "buy_to_open"
    limit_price: float
    notional_usd: float
    client_order_id: str

    @model_validator(mode="after")
    def _plan_is_internally_consistent(self) -> OrderPlan:
        if self.qty < 1:
            raise ValueError(f"qty must be at least 1, got {self.qty}")
        if self.limit_price <= 0:
            raise ValueError(f"limit_price must be positive, got {self.limit_price}")
        expected = round(self.limit_price * 100 * self.qty, 2)
        if round(self.notional_usd, 2) != expected:
            raise ValueError(
                f"notional_usd={self.notional_usd} does not equal limit_price * 100 * qty = {expected}"
            )
        if not self.client_order_id:
            raise ValueError("client_order_id must not be empty")
        allowed = {("buy", "buy_to_open"), ("sell", "sell_to_close")}
        if (self.side, self.position_intent) not in allowed:
            raise ValueError(
                f"side {self.side!r} with intent {self.position_intent!r} is not an allowed order shape"
            )
        return self


RiskReason = Literal[
    # entry
    "no_contract",
    "pending_order_conflict",
    "duplicate_symbol",
    "max_positions",
    "market_closed",
    "too_close_to_close",
    "unacceptable_quote",
    "unknown_buying_power",
    "premium_over_cap",
    "total_premium_over_cap",
    "insufficient_options_buying_power",
    # exit
    "no_position",
    "position_mismatch",
]


class RiskDecision(Observation):
    """Whether an order may be built, and the plan if so.

    ``approved`` carries a plan and no reason; a refusal carries a reason and
    no plan. The two must agree, so a caller can never submit a refusal.
    """

    approved: bool
    reason: RiskReason | None = None
    plan: OrderPlan | None = None

    @model_validator(mode="after")
    def _approval_agrees_with_its_fields(self) -> RiskDecision:
        if self.approved and (self.plan is None or self.reason is not None):
            raise ValueError("an approved decision must carry a plan and no reason")
        if not self.approved and (self.plan is not None or self.reason is None):
            raise ValueError("a refused decision must carry a reason and no plan")
        return self


class OrderReceipt(Observation):
    """What Alpaca said about one submitted order. Never the request itself.

    ``submitted`` is true only when Alpaca returned an order. Fill fields come
    from the single read-back after submission and may still be zero/None:
    a paper fill is simulated and is not required for the cycle to count.
    ``error`` names only an exception type, never upstream text.
    """

    submitted: bool
    order_id: str | None = None
    client_order_id: str | None = None
    status: str | None = None
    submitted_at: UtcDatetime | None = None
    filled_qty: float | None = None
    filled_avg_price: float | None = None
    error: str | None = None


ActionKind = Literal["close", "open"]
ActionOutcome = Literal["planned", "submitted", "rejected", "no_contract", "error"]


class ActionResult(Observation):
    """One requested action (close this symbol, or open one new contract) and
    everything deterministic code did with it. An action's failure never
    touches another action: each carries its own outcome and error.
    """

    kind: ActionKind
    symbol: str
    direction: EntryDirection | None = None
    selection: SelectionResult | None = None
    execution_state: ExecutionState | None = None
    risk: RiskDecision | None = None
    receipt: OrderReceipt | None = None
    outcome: ActionOutcome
    error: str | None = None

    @model_validator(mode="after")
    def _outcome_agrees_with_its_fields(self) -> ActionResult:
        if (self.outcome == "error") != (self.error is not None):
            raise ValueError("outcome 'error' must carry an error message, and only then")
        if self.outcome == "submitted" and not (self.receipt is not None and self.receipt.submitted):
            raise ValueError("outcome 'submitted' requires a receipt with submitted=True")
        if self.outcome == "planned" and not (self.risk is not None and self.risk.approved):
            raise ValueError("outcome 'planned' requires an approved risk decision")
        if self.outcome == "rejected" and not (self.risk is not None and not self.risk.approved):
            raise ValueError("outcome 'rejected' requires a refused risk decision")
        if self.outcome == "no_contract" and (self.selection is None or self.selection.status == "selected"):
            raise ValueError("outcome 'no_contract' requires a selection that chose nothing")
        if self.outcome in ("planned", "rejected", "no_contract") and self.receipt is not None:
            raise ValueError(f"outcome {self.outcome!r} cannot carry an order receipt")
        return self


RunMode = Literal["dry_run", "execute"]
CycleOutcome = Literal["wait", "hold", "planned", "submitted", "rejected", "error"]


class CycleRecord(Observation):
    """One line of the cycle journal: everything one cycle saw, decided and did.

    ``evidence`` is the complete briefing the model received (features, news,
    gates, portfolio with memos), so a line can be replayed. ``outcome`` is
    the aggregate of ``actions``: ``submitted`` if any order went out,
    ``planned`` if any was approved in a dry run, ``rejected`` if actions were
    requested and none got through, ``hold`` if positions exist and nothing
    was requested, ``wait`` if the account is flat and nothing was requested,
    ``error`` only when the cycle itself failed before its actions.
    """

    cycle_id: str
    started_at: UtcDatetime
    finished_at: UtcDatetime
    mode: RunMode
    outcome: CycleOutcome
    forced: str | None = None
    evidence: EvidencePacket | None = None
    decision: PortfolioDecision | None = None
    actions: tuple[ActionResult, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def _outcome_agrees_with_its_fields(self) -> CycleRecord:
        if (self.outcome == "error") != (self.error is not None):
            raise ValueError("outcome 'error' must carry an error message, and only then")
        outcomes = {a.outcome for a in self.actions}
        if self.outcome == "submitted" and "submitted" not in outcomes:
            raise ValueError("outcome 'submitted' requires a submitted action")
        if self.outcome == "planned" and ("planned" not in outcomes or "submitted" in outcomes):
            raise ValueError("outcome 'planned' requires a planned action and no submitted one")
        if self.outcome in ("wait", "hold") and self.actions:
            raise ValueError(f"outcome {self.outcome!r} cannot carry actions")
        return self
