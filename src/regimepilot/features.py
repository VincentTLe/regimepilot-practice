"""Deterministic feature calculation: normalized bars in, one FeaturePacket out.

This module is pure. It performs no network call, imports no vendor SDK, and
holds no client. Everything it needs arrives as an argument, so a FeaturePacket
can be reproduced exactly from the same inputs on any machine, at any time.

Two rules shape every calculation here:

* The feature layer never fabricates market information. A missing, stale or
  invalid input makes the feature that depends on it ``None`` -- never a
  substitute value, never a shorter horizon quietly standing in for a longer
  one, and never an older bar standing in for the newest one.
* One regular session only. A bar belongs to this observation if it falls
  between 09:30 New York and the session's actual close on the session that
  contains ``observed_at`` -- 16:00 normally, 13:00 on a half day. Pre-market,
  post-close and previous-session bars are dropped before any arithmetic, so
  yesterday can never pad today's history and an early close really ends it.

A minute bar is stamped with the *left* edge of its interval: the bar labelled
09:59 covers 09:59:00 (inclusive) to 10:00:00 (exclusive). It is therefore
complete only once ``observed_at`` has reached 10:00:00, and its age is
measured from 09:59:00.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from regimepilot.models import UNDERLYING_SYMBOL, Observation, OhlcvBar, UtcDatetime

# The regular US equity session, which is the only window this phase looks at.
MARKET_TIMEZONE = ZoneInfo("America/New_York")
REGULAR_SESSION_OPEN = time(9, 30)
REGULAR_SESSION_CLOSE = time(16, 0)

# A one-minute bar is complete once this many seconds have passed since it began.
MINUTE_BAR_SECONDS = 60
ONE_MINUTE = timedelta(minutes=1)

RETURN_15M_MINUTES = 15
RETURN_60M_MINUTES = 60

# Thirty log returns need thirty-one closes.
REALIZED_VOL_RETURNS = 30
REALIZED_VOL_CLOSES = REALIZED_VOL_RETURNS + 1

BASIS_POINTS = 10_000

# The only stock feed this phase reads. Stated as a plain string because this
# module may not import a vendor SDK; history.py asserts the two agree.
DATA_FEED_IEX = "iex"


class FeaturePacket(Observation):
    """Five deterministic signal features, plus the context needed to judge them.

    Frozen and closed to extra fields: a packet records what was derivable at
    ``observed_at``. Nothing else may be attached to it later, and no indicator
    beyond the five agreed signals may creep in.

    Every feature is ``None`` when its inputs were missing or insufficient. A
    null here means "not knowable from this observation", never "zero".
    """

    symbol: str = UNDERLYING_SYMBOL
    observed_at: UtcDatetime

    # Context and data quality. Reported, never used as a signal in this phase.
    market_is_open: bool | None = None
    data_feed: str = DATA_FEED_IEX
    minutes_since_open: float | None = None
    minutes_to_close: float | None = None
    spread_bps: float | None = None
    bar_age_seconds: float | None = None

    # The five signal features. All are decimal returns, none is annualized.
    return_15m: float | None = None
    return_60m: float | None = None
    return_since_open: float | None = None
    overnight_gap_pct: float | None = None
    realized_vol_30m: float | None = None


def to_utc(value: datetime) -> datetime:
    """Normalize a timestamp to UTC, treating a naive value as already UTC.

    Deliberately mirrors the rule in ``models`` rather than importing a private
    helper from it, so this module stays free-standing. Public because the
    market-data layer needs the same rule before it can build a request window.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_price(value: float | None) -> bool:
    """A price is usable only if it is a real, strictly positive number."""
    return value is not None and math.isfinite(value) and value > 0


def session_date_of(observed_at: datetime) -> date:
    """The New York calendar date of the session that contains ``observed_at``."""
    return to_utc(observed_at).astimezone(MARKET_TIMEZONE).date()


def session_bounds(session_day: date) -> tuple[datetime, datetime]:
    """The UTC instants at which ``session_day`` opens and closes.

    Built from local New York times so the answer stays correct across daylight
    saving changes.
    """
    opens = datetime.combine(session_day, REGULAR_SESSION_OPEN, tzinfo=MARKET_TIMEZONE)
    closes = datetime.combine(session_day, REGULAR_SESSION_CLOSE, tzinfo=MARKET_TIMEZONE)
    return opens.astimezone(timezone.utc), closes.astimezone(timezone.utc)


def _session_close(session_close_at: datetime | None, session_day: date) -> datetime | None:
    """This session's real close time, or ``None`` if the value is not this session's.

    The market clock reports the *next* close, which is this session's only
    while the market is open. Checking the New York date keeps a next-day value
    from being reported as time remaining in today's session.
    """
    if session_close_at is None:
        return None
    closes_at = to_utc(session_close_at)
    if closes_at.astimezone(MARKET_TIMEZONE).date() != session_day:
        return None
    return closes_at


def session_minute_bars(
    bars: Sequence[OhlcvBar],
    *,
    observed_at: datetime,
    session_day: date | None = None,
    session_close_at: datetime | None = None,
) -> list[OhlcvBar]:
    """The completed regular-session bars of one session, sorted and de-duplicated.

    The session runs from 09:30 New York to ``session_close_at``, which is when
    this session actually closes according to the market clock. On a normal day
    that is 16:00; on a half day it is 13:00, and a bar at or after 13:00 is
    then outside the session and must not reach any feature. A missing or
    out-of-session close time falls back to the 16:00 regular close, which is
    the right default for a session the clock cannot currently describe.

    Nothing assumes the feed declines to return post-close bars; they are
    filtered here whether or not they arrive.

    Dropped here, and therefore invisible to every calculation below:

    * bars with no timestamp,
    * bars outside the session window -- which is one test covering pre-market,
      after-hours, post-early-close, and any other trading day at once,
    * the bar of the currently forming minute.

    Duplicate timestamps resolve deterministically to the **last** bar supplied
    for that timestamp, so a corrected bar supersedes the one it corrects.
    """
    observed_at = to_utc(observed_at)
    session_day = session_day or session_date_of(observed_at)
    opens, regular_close = session_bounds(session_day)
    closes = _session_close(session_close_at, session_day) or regular_close
    complete_by = observed_at - timedelta(seconds=MINUTE_BAR_SECONDS)

    kept = [
        bar
        for bar in bars
        if bar.timestamp is not None
        and opens <= bar.timestamp < closes
        and bar.timestamp <= complete_by
    ]

    # sorted() is stable, so equal timestamps keep the order they were supplied
    # in and the later assignment below wins.
    latest_per_stamp: dict[datetime, OhlcvBar] = {}
    for bar in sorted(kept, key=lambda bar: bar.timestamp):
        latest_per_stamp[bar.timestamp] = bar
    return list(latest_per_stamp.values())


def previous_session_close(
    daily_bars: Sequence[OhlcvBar], session_day: date
) -> float | None:
    """The regular-session close of the newest session before ``session_day``.

    A daily bar is stamped 00:00 New York on its own session date, and its close
    is the official regular-session close, so it is the right source for the
    overnight gap. The current session's own daily bar is excluded: it is today,
    not "previous".
    """
    priors = [
        bar
        for bar in daily_bars
        if bar.timestamp is not None
        and bar.timestamp.astimezone(MARKET_TIMEZONE).date() < session_day
        and _is_price(bar.close)
    ]
    if not priors:
        return None
    return max(priors, key=lambda bar: bar.timestamp).close


def spread_bps(bid: float | None, ask: float | None) -> float | None:
    """Top-of-book spread in basis points, or ``None`` if the quote is unusable.

    Requires a bid, an ask, ``bid > 0``, ``ask >= bid`` and a positive midpoint.
    A locked book (``bid == ask``) is a real zero spread, not an error.
    """
    if bid is None or ask is None:
        return None
    if not (math.isfinite(bid) and math.isfinite(ask)):
        return None
    if bid <= 0 or ask < bid:
        return None

    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return (ask - bid) / mid * BASIS_POINTS


def quote_age_seconds(quote_at: datetime | None, reference: datetime) -> float | None:
    """How old a quote stamped ``quote_at`` was at ``reference``, in seconds.

    ``None`` when there is no quote. A stamp after the reference reads as 0.0
    rather than a negative age; whether such a stamp is acceptable is the
    caller's rule to state, not this function's to hide.
    """
    if quote_at is None:
        return None
    return max(0.0, (to_utc(reference) - to_utc(quote_at)).total_seconds())


def realized_volatility(bars: Sequence[OhlcvBar]) -> float | None:
    """Root sum of squared one-minute log returns over the most recent window.

    ``sqrt(sum(ln(c_i / c_i-1) ** 2))`` over the last thirty returns.

    Returned raw and unscaled: not annualized, not multiplied by sqrt(252),
    sqrt(390) or 100. Fewer than thirty-one closes yields ``None``, because a
    shorter window would be a different measurement wearing this one's name.

    The thirty-one closes must sit on thirty-one *consecutive* minutes. A feed
    that emits no bar for a minute -- which happens when a minute had no
    eligible trade -- would otherwise leave a two-minute move standing in the
    sum as though it were a one-minute return, quietly inflating the result.
    Any gap inside the window yields ``None`` rather than that.
    """
    if len(bars) < REALIZED_VOL_CLOSES:
        return None

    window = bars[-REALIZED_VOL_CLOSES:]
    steps = list(zip(window, window[1:]))
    if any(later.timestamp - earlier.timestamp != ONE_MINUTE for earlier, later in steps):
        return None

    total = sum(math.log(later.close / earlier.close) ** 2 for earlier, later in steps)
    return math.sqrt(total)


def _ratio_change(numerator: float | None, denominator: float | None) -> float | None:
    """``numerator / denominator - 1``, or ``None`` unless both are real prices."""
    if not (_is_price(numerator) and _is_price(denominator)):
        return None
    return numerator / denominator - 1


def build_feature_packet(
    *,
    observed_at: datetime,
    minute_bars: Sequence[OhlcvBar] = (),
    daily_bars: Sequence[OhlcvBar] = (),
    bid: float | None = None,
    ask: float | None = None,
    market_is_open: bool | None = None,
    session_close_at: datetime | None = None,
    symbol: str = UNDERLYING_SYMBOL,
    data_feed: str = DATA_FEED_IEX,
) -> FeaturePacket:
    """Derive one FeaturePacket. Pure: same inputs, same packet, no I/O.

    ``minute_bars`` may arrive unsorted, duplicated, or padded with bars from
    other sessions and other hours; all of that is filtered before any
    arithmetic. Supplying too few bars is not an error -- the features that
    cannot be derived come back ``None``.

    ``session_close_at`` is when this session actually closes, taken from the
    market clock rather than assumed. Half days close at 13:00 New York, so a
    hardcoded 16:00 would both over-report ``minutes_to_close`` by three hours
    and let post-close bars into the returns. It bounds the session for every
    feature. ``minutes_to_close`` and ``minutes_since_open`` are ``None``
    without it -- an unknown close time is not a 16:00 one -- while the bar
    filter falls back to the 16:00 regular close.
    """
    observed_at = to_utc(observed_at)
    session_day = session_date_of(observed_at)
    opens, _ = session_bounds(session_day)
    closes_at = _session_close(session_close_at, session_day)

    bars = session_minute_bars(
        minute_bars,
        observed_at=observed_at,
        session_day=session_day,
        session_close_at=session_close_at,
    )

    # The latest completed regular-session bar, whatever its close turned out
    # to be: this is what bar_age_seconds describes.
    latest_bar = bars[-1] if bars else None

    # Only bars carrying a real price take part in the return calculations.
    priced = [bar for bar in bars if _is_price(bar.close)]
    close_at_stamp = {bar.timestamp: bar.close for bar in priced}

    # Every intraday feature is measured from the newest completed bar or from
    # nothing at all. If that bar arrived with an unusable close, an older bar
    # must not take its place: the arithmetic would still succeed and would
    # report a move that ended minutes ago as the current one, with no field
    # anywhere in the packet showing that the observation had slipped backwards.
    anchor = latest_bar if latest_bar is not None and _is_price(latest_bar.close) else None
    latest_close = anchor.close if anchor is not None else None
    latest_stamp = anchor.timestamp if anchor is not None else None

    def close_minutes_before(minutes: int) -> float | None:
        """The close of the bar exactly ``minutes`` before the latest one.

        An exact timestamp, never a nearest-neighbour: if that minute produced
        no usable bar there is no such close, and the feature goes null rather
        than silently reporting a different horizon.
        """
        if latest_stamp is None:
            return None
        return close_at_stamp.get(latest_stamp - timedelta(minutes=minutes))

    # The open of the 09:30 bar, which is the regular session open. A
    # pre-market print is never allowed to stand in for it.
    opening_bar = next((bar for bar in bars if bar.timestamp == opens), None)
    session_open = opening_bar.open if opening_bar is not None else None
    if not _is_price(session_open):
        session_open = None

    # The volatility window has to end on the anchor for the same reason, so a
    # missing anchor makes it null rather than a measurement of an older window.
    realized_vol_30m = realized_volatility(priced) if anchor is not None else None

    bar_age_seconds = (
        None
        if latest_bar is None
        else (observed_at - latest_bar.timestamp).total_seconds()
    )

    # Session minutes are reported only when the market is open and the clock
    # agrees with the session window. Anything else is null, not a guess -- and
    # an unknown close time makes the whole window unknowable.
    minutes_since_open = minutes_to_close = None
    if market_is_open is True and closes_at is not None and opens <= observed_at <= closes_at:
        minutes_since_open = (observed_at - opens).total_seconds() / 60
        minutes_to_close = (closes_at - observed_at).total_seconds() / 60

    return FeaturePacket(
        symbol=symbol,
        observed_at=observed_at,
        market_is_open=market_is_open,
        data_feed=data_feed,
        minutes_since_open=minutes_since_open,
        minutes_to_close=minutes_to_close,
        spread_bps=spread_bps(bid, ask),
        bar_age_seconds=bar_age_seconds,
        return_15m=_ratio_change(latest_close, close_minutes_before(RETURN_15M_MINUTES)),
        return_60m=_ratio_change(latest_close, close_minutes_before(RETURN_60M_MINUTES)),
        return_since_open=_ratio_change(latest_close, session_open),
        overnight_gap_pct=_ratio_change(
            session_open, previous_session_close(daily_bars, session_day)
        ),
        realized_vol_30m=realized_vol_30m,
    )


def _signal(value: float | None) -> str:
    """A decimal return, shown with enough digits to be checkable by hand."""
    return "null" if value is None else f"{value:+.6f}"


def _context(value: float | None, digits: int = 2) -> str:
    return "null" if value is None else f"{value:,.{digits}f}"


def format_summary(packet: FeaturePacket) -> str:
    """A compact, honest summary. Nulls are printed as nulls, never as zeros."""
    state = (
        "unknown"
        if packet.market_is_open is None
        else ("OPEN" if packet.market_is_open else "CLOSED")
    )
    stamp = packet.observed_at.astimezone(MARKET_TIMEZONE)

    return "\n".join(
        [
            f"RegimePilot features  {packet.symbol}  @ "
            f"{stamp.strftime('%Y-%m-%d %H:%M:%S')} ET  "
            f"({packet.observed_at.strftime('%H:%M:%SZ')})",
            f"  {'market':<20} {state}   feed {packet.data_feed}",
            f"  {'session minutes':<20} since open {_context(packet.minutes_since_open, 1)}"
            f"   to close {_context(packet.minutes_to_close, 1)}",
            f"  {'data quality':<20} spread {_context(packet.spread_bps)} bps"
            f"   latest bar age {_context(packet.bar_age_seconds, 1)} s",
            f"  {'return_15m':<20} {_signal(packet.return_15m)}",
            f"  {'return_60m':<20} {_signal(packet.return_60m)}",
            f"  {'return_since_open':<20} {_signal(packet.return_since_open)}",
            f"  {'overnight_gap_pct':<20} {_signal(packet.overnight_gap_pct)}",
            f"  {'realized_vol_30m':<20} "
            f"{'null' if packet.realized_vol_30m is None else f'{packet.realized_vol_30m:.6f}'}"
            "   (raw decimal, not annualized)",
        ]
    )
