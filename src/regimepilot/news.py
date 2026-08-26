"""Read-only Alpaca news fetch and filter for Phase 3B.

SAFETY: this module is read-only by design. It contains no function that
submits, cancels or replaces an order, and none that closes or exercises a
position. Do not add one here.

Two failure modes are kept strictly apart:

* A call succeeds but carries no articles -> ``available=True`` with zero items.
* A call fails -> ``NewsError``. A consumer may catch that and substitute an
  unavailable packet rather than invent headlines.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

from regimepilot.config import ConfigError, Settings, load_settings
from regimepilot.console import tolerant_console
from regimepilot.features import to_utc
from regimepilot.models import UNDERLYING_SYMBOL, NewsItem, NewsPacket

UNDERLYING = UNDERLYING_SYMBOL

NEWS_LOOKBACK_HOURS = 6
NEWS_FETCH_LIMIT = 50
MAX_NEWS_ITEMS = 5

MACRO_KEYWORDS = frozenset(
    {"fed", "cpi", "rate", "rates", "inflation", "jobs", "fomc", "gdp", "treasury"}
)

_WORD_BOUNDARY = re.compile(r"[a-z0-9]+")

__all__ = [
    "NewsError",
    "build_news_client",
    "build_news_packet",
    "fetch_news",
    "format_summary",
    "headline_is_relevant",
    "observe_news",
    "main",
]


class NewsError(RuntimeError):
    """A read-only news request could not be completed."""


def build_news_client(settings: Settings) -> NewsClient:
    """Create the read-only Alpaca news client from paper credentials."""
    if not settings.paper:
        raise ConfigError("Refusing to build news client: paper trading is not enabled.")

    return NewsClient(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
    )


def _guarded(label: str, call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception as error:  # noqa: BLE001 - deliberately uniform
        raise NewsError(f"failed to read {label}: {type(error).__name__}") from None


def _normalize_symbols(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _normalize_article(row: Any, *, observed_at: datetime) -> NewsItem | None:
    headline = str(getattr(row, "headline", "") or "").strip()
    if not headline:
        return None

    article_id = getattr(row, "id", None)
    if article_id is None:
        return None

    created_at = getattr(row, "created_at", None)
    if not isinstance(created_at, datetime):
        return None

    created_at = to_utc(created_at)
    age_minutes = max(0.0, (observed_at - created_at).total_seconds() / 60)

    summary = str(getattr(row, "summary", "") or headline).strip()
    source = getattr(row, "source", None)
    source_text = None if source is None else str(source).strip() or None

    return NewsItem(
        id=int(article_id),
        headline=headline,
        summary=summary,
        age_minutes=age_minutes,
        symbols=_normalize_symbols(getattr(row, "symbols", None)),
        source=source_text,
    )


def headline_is_relevant(
    headline: str,
    symbols: Sequence[str],
    *,
    underlying: str = UNDERLYING,
) -> bool:
    """True when the article mentions the underlying or a macro keyword."""
    upper_symbols = {symbol.upper() for symbol in symbols}
    if underlying.upper() in upper_symbols:
        return True

    words = set(_WORD_BOUNDARY.findall(headline.lower()))
    return bool(words & MACRO_KEYWORDS)


def _rows(response: Any) -> list[Any]:
    """Pull the article list out of the SDK reply.

    ``NewsClient.get_news`` returns a ``NewsSet`` (a ``BaseDataSet``) whose
    payload lives under ``.data["news"]``; the object has no ``.news``
    attribute. A raw dict reply (``raw_data=True``) carries the same ``"news"``
    key at the top level. Mirrors ``history._rows`` so both Alpaca boundaries
    unwrap a reply the same way.
    """
    data = response if isinstance(response, dict) else getattr(response, "data", None)
    return list((data or {}).get("news") or [])


def fetch_news(
    news_client: Any,
    *,
    observed_at: datetime | None = None,
    symbol: str = UNDERLYING,
    lookback_hours: int = NEWS_LOOKBACK_HOURS,
    limit: int = NEWS_FETCH_LIMIT,
) -> list[Any]:
    """Fetch raw SDK ``News`` rows from Alpaca for one symbol window."""
    observed_at = to_utc(observed_at) if observed_at else datetime.now(timezone.utc)
    start = observed_at - timedelta(hours=lookback_hours)

    request = NewsRequest(
        symbols=symbol,
        start=start,
        end=observed_at,
        sort="desc",
        limit=limit,
        include_content=False,
    )
    response = _guarded(f"{symbol} news", lambda: news_client.get_news(request))
    return _rows(response)


def build_news_packet(
    raw_articles: Sequence[Any],
    *,
    observed_at: datetime,
    symbol: str = UNDERLYING,
    available: bool = True,
) -> NewsPacket:
    """Filter, sort and cap raw news rows into one NewsPacket."""
    observed_at = to_utc(observed_at)
    normalized: list[NewsItem] = []

    for row in raw_articles:
        item = _normalize_article(row, observed_at=observed_at)
        if item is None:
            continue
        if not headline_is_relevant(item.headline, item.symbols, underlying=symbol):
            continue
        normalized.append(item)

    normalized.sort(key=lambda item: item.age_minutes)
    kept = tuple(normalized[:MAX_NEWS_ITEMS])

    return NewsPacket(
        observed_at=observed_at,
        available=available,
        item_count=len(kept),
        items=kept,
    )


def unavailable_news_packet(*, observed_at: datetime) -> NewsPacket:
    """Explicit unavailable packet when the news request failed upstream."""
    return NewsPacket(
        observed_at=to_utc(observed_at),
        available=False,
        item_count=0,
        items=(),
    )


def observe_news(
    news_client: Any,
    *,
    now: datetime | None = None,
    symbol: str = UNDERLYING,
) -> NewsPacket:
    """Fetch and normalize one NewsPacket. Raises ``NewsError`` on request failure."""
    observed_at = to_utc(now) if now else datetime.now(timezone.utc)
    raw_articles = fetch_news(news_client, observed_at=observed_at, symbol=symbol)
    return build_news_packet(raw_articles, observed_at=observed_at, symbol=symbol)


def format_summary(packet: NewsPacket) -> str:
    """Compact human-readable summary."""
    if not packet.available:
        return f"RegimePilot news  unavailable  @ {packet.observed_at.strftime('%H:%M:%SZ')}"

    lines = [
        f"RegimePilot news  {packet.item_count} item(s)"
        f"  @ {packet.observed_at.strftime('%H:%M:%SZ')}",
    ]
    for item in packet.items:
        age = f"{item.age_minutes:.0f}m"
        lines.append(f"  [{age}] {item.headline}")
    if packet.item_count == 0:
        lines.append("  (no relevant headlines in window)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a NewsPacket summary, or the packet itself with ``--json``."""
    tolerant_console()
    arguments = list(sys.argv[1:] if argv is None else argv)

    try:
        settings = load_settings()
        news_client = build_news_client(settings)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    try:
        packet = observe_news(news_client)
    except NewsError as error:
        print(f"news read failed: {error}", file=sys.stderr)
        return 1

    if "--json" in arguments:
        print(json.dumps(json.loads(packet.model_dump_json()), indent=2))
    else:
        print(format_summary(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
