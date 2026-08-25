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

from pydantic import AfterValidator, BaseModel, ConfigDict

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
    vol_regime: str = "unknown"
    session_phase: str = "unknown"


class NewsEvidence(Observation):
    """News slice of an evidence packet."""

    available: bool = False
    item_count: int = 0
    items: tuple[NewsItem, ...] = ()


class AccountHint(Observation):
    """Minimal account state needed before Phase 5 risk sizing."""

    has_open_option_position: bool = False


class EvidencePacket(Observation):
    """One normalized briefing for LLM direction reasoning."""

    observed_at: UtcDatetime
    symbol: str = UNDERLYING_SYMBOL
    gates: GatesEvidence
    underlying: UnderlyingEvidence
    news: NewsEvidence
    account: AccountHint


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
