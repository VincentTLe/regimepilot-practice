"""News tests: the Alpaca boundary of Phase 3B.

The fake client returns the SDK's own ``NewsSet`` wrapper, built offline from
API-shaped dicts, so every test here sees the exact response shape production
sees -- articles under ``response.data["news"]`` -- never a hand-rolled stand-in.
"""

import traceback
from datetime import datetime, timezone

import pytest
from alpaca.data.models.news import News, NewsSet
from alpaca.data.requests import NewsRequest

from regimepilot import news as news_module
from regimepilot.models import NewsPacket
from regimepilot.news import (
    NewsError,
    build_news_packet,
    fetch_news,
    headline_is_relevant,
    observe_news,
    unavailable_news_packet,
)

OBSERVED_AT = datetime(2026, 8, 24, 14, 0, tzinfo=timezone.utc)

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"


def article(
    *,
    article_id,
    headline,
    summary=None,
    symbols=None,
    source="benzinga",
    created_at=None,
    content="<p>secret html</p>",
):
    """One article exactly as the /v1beta1/news endpoint returns it."""
    stamp = (created_at or datetime(2026, 8, 24, 13, 30, tzinfo=timezone.utc)).isoformat()
    return {
        "id": article_id,
        "headline": headline,
        "summary": summary or headline,
        "symbols": symbols or ["SPY"],
        "source": source,
        "created_at": stamp,
        "updated_at": stamp,
        "author": "desk",
        "content": content,
        "url": "https://example.com/news",
        "images": [],
    }


def sdk_news(**kwargs):
    """The SDK ``News`` model for one article, exactly as ``NewsSet`` builds it."""
    return News(raw_data=article(**kwargs))


def sdk_response(articles):
    """The real SDK wrapper that ``NewsClient.get_news`` returns, built offline."""
    return NewsSet({"news": list(articles), "next_page_token": None})


class FakeNewsClient:
    def __init__(self, articles=(), *, fail=False):
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY
        self.articles = list(articles)
        self.fail = fail
        self.last_request = None

    def get_news(self, request):
        self.last_request = request
        if self.fail:
            raise RuntimeError(f"401 unauthorized for key={API_KEY} secret={SECRET_KEY}")
        return sdk_response(self.articles)


def test_fetch_news_reads_articles_from_the_real_sdk_newsset():
    """Regression: NewsSet keeps articles under .data["news"] and has no .news attribute."""
    response = sdk_response([article(article_id=1, headline="SPY steady")])
    assert not hasattr(response, "news")
    assert len(response.data["news"]) == 1

    client = FakeNewsClient([article(article_id=1, headline="SPY steady")])
    rows = fetch_news(client, observed_at=OBSERVED_AT)

    assert len(rows) == 1
    assert isinstance(rows[0], News)
    assert rows[0].headline == "SPY steady"


def test_observe_news_carries_the_articles_the_sdk_wrapper_returned():
    client = FakeNewsClient(
        [
            article(
                article_id=7,
                headline="SPY update",
                created_at=datetime(2026, 8, 24, 13, 50, tzinfo=timezone.utc),
            ),
            article(
                article_id=8,
                headline="Fed holds rates steady",
                symbols=["QQQ"],
                created_at=datetime(2026, 8, 24, 13, 40, tzinfo=timezone.utc),
            ),
        ]
    )

    packet = observe_news(client, now=OBSERVED_AT)

    assert packet.available is True
    assert packet.item_count == 2
    assert [item.id for item in packet.items] == [7, 8]
    assert [item.age_minutes for item in packet.items] == [10.0, 20.0]


def test_headline_is_relevant_for_spy_symbol_or_macro_keyword():
    assert headline_is_relevant("SPY rises", ["SPY"]) is True
    assert headline_is_relevant("Fed holds rates steady", ["QQQ"]) is True
    assert headline_is_relevant("Random company beats earnings", ["AAPL"]) is False


def test_build_news_packet_filters_sorts_and_caps_items():
    articles = [
        sdk_news(article_id=1, headline="Old SPY move", created_at=datetime(2026, 8, 24, 10, 0, tzinfo=timezone.utc)),
        sdk_news(article_id=2, headline="Fresh SPY move", created_at=datetime(2026, 8, 24, 13, 45, tzinfo=timezone.utc)),
        sdk_news(article_id=3, headline="Irrelevant headline", symbols=["AAPL"], created_at=datetime(2026, 8, 24, 13, 50, tzinfo=timezone.utc)),
        sdk_news(article_id=4, headline="Fed comments lift index ETFs", symbols=["QQQ"], created_at=datetime(2026, 8, 24, 13, 55, tzinfo=timezone.utc)),
        sdk_news(article_id=5, headline="SPY five", created_at=datetime(2026, 8, 24, 13, 40, tzinfo=timezone.utc)),
        sdk_news(article_id=6, headline="SPY six", created_at=datetime(2026, 8, 24, 13, 35, tzinfo=timezone.utc)),
    ]

    packet = build_news_packet(articles, observed_at=OBSERVED_AT)

    assert packet.available is True
    assert packet.item_count == 5
    assert [item.id for item in packet.items] == [4, 2, 5, 6, 1]
    assert all("content" not in item.model_dump() for item in packet.items)


def test_fetch_news_sends_the_expected_request_window():
    client = FakeNewsClient([article(article_id=1, headline="SPY steady")])
    rows = fetch_news(client, observed_at=OBSERVED_AT, symbol="SPY")

    assert len(rows) == 1
    assert isinstance(client.last_request, NewsRequest)
    assert client.last_request.symbols == "SPY"
    assert client.last_request.include_content is False
    assert client.last_request.limit == 50


def test_observe_news_raises_a_credential_safe_error_on_failure():
    client = FakeNewsClient(fail=True)

    with pytest.raises(NewsError) as excinfo:
        observe_news(client, now=OBSERVED_AT)

    assert "RuntimeError" in str(excinfo.value)
    assert API_KEY not in str(excinfo.value)
    assert SECRET_KEY not in str(excinfo.value)
    assert API_KEY not in traceback.format_exc()


def test_unavailable_news_packet_is_explicit():
    packet = unavailable_news_packet(observed_at=OBSERVED_AT)

    assert packet.available is False
    assert packet.item_count == 0
    assert packet.items == ()


def test_news_packet_is_frozen_and_closed_to_stray_fields():
    packet = build_news_packet([], observed_at=OBSERVED_AT)

    with pytest.raises(Exception):
        packet.available = True
    with pytest.raises(Exception):
        NewsPacket(**{**packet.model_dump(), "extra": True})


def test_the_news_module_exposes_no_trading_or_execution_helper():
    forbidden = (
        "submit", "cancel", "replace", "close_position", "close_all", "exercise",
        "order", "buy_call", "buy_put", "position", "size", "risk", "decide",
    )
    offenders = [
        name for name in dir(news_module) if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []


def test_a_successful_observation_serializes_without_credentials():
    client = FakeNewsClient([article(article_id=7, headline="SPY update")])
    packet = observe_news(client, now=OBSERVED_AT)
    blob = packet.model_dump_json()

    assert API_KEY not in blob
    assert SECRET_KEY not in blob
    assert NewsPacket.model_validate_json(blob) == packet
