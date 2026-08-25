"""Read-only Alpaca connectivity check for SPY.

SAFETY: this module is read-only by design. It contains no function that
submits, cancels or replaces an order, and none that closes or exercises a
position. Do not add one here.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus
from alpaca.trading.requests import GetOptionContractsRequest

from regimepilot.config import ConfigError, Settings, load_settings

UNDERLYING = "SPY"
BAR_LOOKBACK_DAYS = 10
MIN_DAYS_TO_EXPIRATION = 3
MAX_DAYS_TO_EXPIRATION = 14

# Option expirations are US market calendar dates, so days-to-expiration has to
# be counted from the market's date rather than from UTC's. Between 00:00 UTC
# and New York midnight the two disagree: UTC is already on the next day while
# the options market is still on the previous one. Spelled out here rather than
# imported, because this module is the one the observer imports from, not the
# other way round.
MARKET_TIMEZONE = ZoneInfo("America/New_York")

# One page is enough to prove connectivity. The counts reported below are
# therefore "contracts returned", not "contracts that exist".
OPTION_CONTRACT_PAGE_LIMIT = 500

CHECKS = ("config", "clock", "account", "spy_bars", "spy_option_contracts")

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


def mask_account_id(account_id: Any) -> str:
    """Show at most the last four characters of an account id."""
    text = "" if account_id is None else str(account_id)
    if not text:
        return "unavailable"
    if len(text) <= 4:
        return "****"
    return "****" + text[-4:]


def empty_report(now: datetime) -> dict[str, Any]:
    """The report skeleton, with every check unrun."""
    return {
        "timestamp": now.astimezone(timezone.utc).isoformat(),
        "market_open": None,
        "account_id_masked": "unavailable",
        "spy_bar_count": 0,
        "spy_option_contract_count": 0,
        "earliest_expiration": None,
        "latest_expiration": None,
        "checks": {name: STATUS_SKIPPED for name in CHECKS},
    }


def _report_failure(check: str, error: BaseException) -> None:
    """Announce a failed check on stderr without echoing the exception message.

    An HTTP client's exception text can quote the request it made, so only the
    exception type is ever surfaced.
    """
    print(f"check '{check}' failed: {type(error).__name__}", file=sys.stderr)


def _expiration_dates(contracts: list[Any]) -> list[str]:
    """Sorted, de-duplicated ISO expiration dates of the returned contracts."""
    dates = set()
    for contract in contracts:
        value = getattr(contract, "expiration_date", None)
        if value is None:
            continue
        dates.add(value.isoformat() if isinstance(value, date) else str(value))
    return sorted(dates)


def _fetch_market_open(trading_client: Any) -> bool:
    return bool(trading_client.get_clock().is_open)


def _fetch_account_id(trading_client: Any) -> str:
    return str(getattr(trading_client.get_account(), "id", ""))


def _fetch_spy_bars(data_client: Any, now: datetime) -> list[Any]:
    request = StockBarsRequest(
        symbol_or_symbols=[UNDERLYING],
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
        start=now - timedelta(days=BAR_LOOKBACK_DAYS),
    )
    barset = data_client.get_stock_bars(request)
    return list(getattr(barset, "data", {}).get(UNDERLYING) or [])


def _market_date(now: datetime) -> date:
    """The New York calendar date of ``now``. A naive value is taken as UTC.

    Converted through a real timezone, never a fixed offset, so the answer stays
    right on both sides of a daylight saving change.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(MARKET_TIMEZONE).date()


def _fetch_spy_option_contracts(trading_client: Any, now: datetime) -> list[Any]:
    today = _market_date(now)
    request = GetOptionContractsRequest(
        underlying_symbols=[UNDERLYING],
        status=AssetStatus.ACTIVE,
        expiration_date_gte=today + timedelta(days=MIN_DAYS_TO_EXPIRATION),
        expiration_date_lte=today + timedelta(days=MAX_DAYS_TO_EXPIRATION),
        limit=OPTION_CONTRACT_PAGE_LIMIT,
    )
    response = trading_client.get_option_contracts(request)
    return list(getattr(response, "option_contracts", None) or [])


def run_smoke_test(
    trading_client: Any,
    data_client: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run every read-only check and return the normalized report.

    Clients are injected so that unit tests can exercise this without a network
    call. Each check is isolated: one failure does not hide the others.
    """
    now = now or datetime.now(timezone.utc)
    report = empty_report(now)
    checks = report["checks"]

    # Clients can only be built from settings that already passed validation.
    checks["config"] = STATUS_OK

    try:
        report["market_open"] = _fetch_market_open(trading_client)
        checks["clock"] = STATUS_OK
    except Exception as error:  # noqa: BLE001 - one bad check must not hide the rest
        checks["clock"] = STATUS_ERROR
        _report_failure("clock", error)

    try:
        report["account_id_masked"] = mask_account_id(_fetch_account_id(trading_client))
        checks["account"] = STATUS_OK
    except Exception as error:  # noqa: BLE001
        checks["account"] = STATUS_ERROR
        _report_failure("account", error)

    try:
        bars = _fetch_spy_bars(data_client, now)
        report["spy_bar_count"] = len(bars)
        checks["spy_bars"] = STATUS_OK if bars else STATUS_EMPTY
    except Exception as error:  # noqa: BLE001
        checks["spy_bars"] = STATUS_ERROR
        _report_failure("spy_bars", error)

    try:
        contracts = _fetch_spy_option_contracts(trading_client, now)
        expirations = _expiration_dates(contracts)
        report["spy_option_contract_count"] = len(contracts)
        report["earliest_expiration"] = expirations[0] if expirations else None
        report["latest_expiration"] = expirations[-1] if expirations else None
        checks["spy_option_contracts"] = STATUS_OK if contracts else STATUS_EMPTY
    except Exception as error:  # noqa: BLE001
        checks["spy_option_contracts"] = STATUS_ERROR
        _report_failure("spy_option_contracts", error)

    return report


def build_clients(settings: Settings) -> tuple[TradingClient, StockHistoricalDataClient]:
    """Create the read-only Alpaca clients.

    ``paper=True`` is hard-coded rather than read from configuration, so no
    environment value can flip this project onto the live endpoint.
    """
    if not settings.paper:
        raise ConfigError("Refusing to build clients: paper trading is not enabled.")

    api_key = settings.alpaca_api_key.get_secret_value()
    secret_key = settings.alpaca_secret_key.get_secret_value()

    trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    return trading_client, data_client


def main() -> int:
    """Print the smoke-test report as JSON. Returns a process exit code."""
    now = datetime.now(timezone.utc)

    try:
        settings = load_settings()
        trading_client, data_client = build_clients(settings)
    except ConfigError as error:
        report = empty_report(now)
        report["checks"]["config"] = STATUS_ERROR
        print(json.dumps(report, indent=2))
        # ConfigError messages are built by us and never contain a credential.
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    report = run_smoke_test(trading_client, data_client, now=now)
    print(json.dumps(report, indent=2))

    ok = all(status in (STATUS_OK, STATUS_EMPTY) for status in report["checks"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
