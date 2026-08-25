"""Smoke-test tests. Alpaca clients are replaced with fakes, so no network
call is ever made."""

import json
from datetime import date, datetime, timezone

from regimepilot.smoke_test import (
    CHECKS,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    mask_account_id,
    run_smoke_test,
)

NOW = datetime(2026, 8, 25, 14, 30, tzinfo=timezone.utc)

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"
ACCOUNT_ID = "11112222-3333-4444-5555-666677778888"

REPORT_KEYS = {
    "timestamp",
    "market_open",
    "account_id_masked",
    "spy_bar_count",
    "spy_option_contract_count",
    "earliest_expiration",
    "latest_expiration",
    "checks",
}


class FakeClock:
    def __init__(self, is_open):
        self.is_open = is_open


class FakeAccount:
    def __init__(self, account_id):
        self.id = account_id


class FakeContract:
    def __init__(self, expiration):
        self.expiration_date = expiration


class FakeBar:
    pass


class FakeBarSet:
    def __init__(self, bars):
        self.data = {"SPY": list(bars)}


class FakeContractsResponse:
    def __init__(self, contracts):
        self.option_contracts = list(contracts)


class FakeTradingClient:
    """Stands in for alpaca TradingClient. Records requests, hits no network."""

    def __init__(self, *, is_open=True, account_id=ACCOUNT_ID, contracts=()):
        self._is_open = is_open
        self._account_id = account_id
        self._contracts = list(contracts)
        self.option_requests = []
        # Deliberately carries credentials so the leak test below is meaningful.
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_clock(self):
        return FakeClock(self._is_open)

    def get_account(self):
        return FakeAccount(self._account_id)

    def get_option_contracts(self, request):
        self.option_requests.append(request)
        return FakeContractsResponse(self._contracts)


class FakeDataClient:
    def __init__(self, bars=()):
        self._bars = list(bars)
        self.bar_requests = []

    def get_stock_bars(self, request):
        self.bar_requests.append(request)
        return FakeBarSet(self._bars)


def test_successful_normalization():
    trading = FakeTradingClient(
        is_open=True,
        contracts=[
            FakeContract(date(2026, 9, 4)),
            FakeContract(date(2026, 8, 28)),
            FakeContract(date(2026, 9, 4)),
        ],
    )
    data = FakeDataClient(bars=[FakeBar(), FakeBar(), FakeBar()])

    report = run_smoke_test(trading, data, now=NOW)

    assert report["timestamp"] == "2026-08-25T14:30:00+00:00"
    assert report["market_open"] is True
    assert report["account_id_masked"] == "****8888"
    assert report["spy_bar_count"] == 3
    assert report["spy_option_contract_count"] == 3
    assert report["earliest_expiration"] == "2026-08-28"
    assert report["latest_expiration"] == "2026-09-04"
    assert report["checks"] == {name: STATUS_OK for name in CHECKS}


def test_report_exposes_only_the_agreed_fields():
    report = run_smoke_test(FakeTradingClient(), FakeDataClient(), now=NOW)
    assert set(report) == REPORT_KEYS
    assert set(report["checks"]) == set(CHECKS)


def test_empty_spy_bars():
    trading = FakeTradingClient(contracts=[FakeContract(date(2026, 9, 1))])
    report = run_smoke_test(trading, FakeDataClient(bars=[]), now=NOW)

    assert report["spy_bar_count"] == 0
    assert report["checks"]["spy_bars"] == STATUS_EMPTY
    # An empty result is not a failure, and it must not affect the other checks.
    assert report["checks"]["spy_option_contracts"] == STATUS_OK


def test_empty_option_contracts():
    report = run_smoke_test(
        FakeTradingClient(contracts=[]), FakeDataClient(bars=[FakeBar()]), now=NOW
    )

    assert report["spy_option_contract_count"] == 0
    assert report["earliest_expiration"] is None
    assert report["latest_expiration"] is None
    assert report["checks"]["spy_option_contracts"] == STATUS_EMPTY
    assert report["checks"]["spy_bars"] == STATUS_OK


def test_option_request_uses_the_three_to_fourteen_day_window():
    trading = FakeTradingClient()
    run_smoke_test(trading, FakeDataClient(), now=NOW)

    request = trading.option_requests[0]
    assert list(request.underlying_symbols) == ["SPY"]
    assert request.expiration_date_gte == date(2026, 8, 28)
    assert request.expiration_date_lte == date(2026, 9, 8)


def test_failed_check_is_isolated_and_does_not_echo_the_exception_message(capsys):
    class BoomTradingClient(FakeTradingClient):
        def get_clock(self):
            raise RuntimeError(f"upstream rejected key {API_KEY}")

    report = run_smoke_test(BoomTradingClient(), FakeDataClient(bars=[FakeBar()]), now=NOW)

    assert report["checks"]["clock"] == STATUS_ERROR
    assert report["market_open"] is None
    assert report["checks"]["account"] == STATUS_OK
    assert report["checks"]["spy_bars"] == STATUS_OK

    stderr = capsys.readouterr().err
    assert "RuntimeError" in stderr
    assert API_KEY not in stderr


def test_secrets_never_appear_in_the_output():
    trading = FakeTradingClient(contracts=[FakeContract(date(2026, 9, 4))])
    serialized = json.dumps(run_smoke_test(trading, FakeDataClient(bars=[FakeBar()]), now=NOW))

    assert API_KEY not in serialized
    assert SECRET_KEY not in serialized
    assert ACCOUNT_ID not in serialized
    assert "****8888" in serialized  # only the masked tail survives


def test_mask_account_id():
    assert mask_account_id("abcdefgh") == "****efgh"
    assert mask_account_id("abcd") == "****"
    assert mask_account_id("") == "unavailable"
    assert mask_account_id(None) == "unavailable"


def test_module_exposes_no_order_or_position_mutation():
    """Phase 1 is read-only. Guard against an execution helper sneaking in."""
    import regimepilot.smoke_test as module

    forbidden = ("submit", "cancel", "replace", "close_position", "close_all", "exercise")
    offenders = [
        name
        for name in dir(module)
        if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []
