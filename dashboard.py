"""Dashboard data export, local serving and optional surge deploy for the loop.

A Python port of surge_artifacts/*/deploy.sh so it runs on Windows without sh.
Every step is best-effort: a failure is logged and reported in the returned
status dict, never raised into the trading loop.

The Alpaca CLI (`alpaca clock / account get / position list`) feeds
cli_snapshot.json — the hackathon requires the project to utilize Alpaca's MCP
server or CLI. Orders still go only through broker.submit_paper_order.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer
import yaml
from loguru import logger

ROOT = Path(__file__).parent
STEP_TIMEOUT = 90  # seconds per export subprocess
CANDLES_TIMEOUT = 120  # export_candles.py fetches bars for every whitelisted symbol
CLI_TIMEOUT = 20
# Only these account fields leave the CLI reply (no keys, no PII beyond the paper account number).
ACCOUNT_FIELDS = ("account_number", "equity", "last_equity", "cash", "buying_power",
                  "status", "options_trading_level")

app = typer.Typer(add_completion=False, no_args_is_help=True)


class DashboardError(Exception):
    pass


def cycles_dir(root: Path = ROOT) -> Path:
    return root / "surge_artifacts" / "paca-cycles"


def candles_dir(root: Path = ROOT) -> Path:
    return root / "surge_artifacts" / "paca-candles"


def _run(args: list[str], timeout: int = STEP_TIMEOUT) -> str:
    """Run a subprocess in the repo root and return stdout.

    Errors name the program and exit code / exception type only — never the
    output, which could carry account details.
    """
    try:
        completed = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,  # never let a tool's colour codes raise a decode error
        )
    except Exception as error:
        raise DashboardError(f"{Path(args[0]).name} failed: {type(error).__name__}") from None
    if completed.returncode != 0:
        raise DashboardError(f"{Path(args[0]).name} exited {completed.returncode}")
    return completed.stdout


def _py(*args: str) -> list[str]:
    """The current interpreter (the uv venv), so exports need no resolver step."""
    return [sys.executable, *args]


def _write_atomic(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def export_account(root: Path = ROOT) -> str:
    _run(_py("cli.py", "account", "--export"))
    src = root / "logs" / "account.json"
    if not src.exists():
        raise DashboardError("account export wrote nothing")
    cycles_dir(root).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, cycles_dir(root) / "account.json")
    return "ok"


def export_pnl(root: Path = ROOT) -> str:
    for name, args in (
        ("positions.json", ["positions", "--json"]),
        ("realized.json", ["realized", "--json", "--days", "30"]),
    ):
        text = _run(_py("pnl.py", *args))
        _write_atomic(cycles_dir(root) / name, text)
    return "ok"


def write_config_json(settings_path: Path, out_path: Path) -> str:
    """settings.yaml minus the llm section (endpoint + model names stay private)."""
    data = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    data.pop("llm", None)
    _write_atomic(out_path, json.dumps(data))
    return "ok"


def copy_journal(root: Path = ROOT) -> str:
    src = root / "logs" / "cycles.jsonl"
    if not src.exists():
        raise DashboardError("no journal yet")
    cycles_dir(root).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, cycles_dir(root) / "cycles.jsonl")
    return "ok"


def cli_snapshot(
    profile: str | None,
    out_path: Path | None = None,
    expected_account_number: str | None = None,
) -> dict:
    """Broker state read through the Alpaca CLI. Falls back to a stub, never raises.

    When `expected_account_number` (the account the engine trades, from the SDK
    export) is given and the CLI profile points at another account, nothing of
    that other account is published: the stub says `account_mismatch`.
    """
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    binary = shutil.which("alpaca")
    if binary is None:
        snapshot: dict = {"source": "sdk", "cli_error": "cli_not_found", "generated_at": generated}
    else:
        suffix = ["--quiet"] + (["-p", profile] if profile else [])
        try:
            clock = json.loads(_run([binary, "clock", *suffix], timeout=CLI_TIMEOUT))
            account = json.loads(_run([binary, "account", "get", *suffix], timeout=CLI_TIMEOUT))
            positions = json.loads(_run([binary, "position", "list", *suffix], timeout=CLI_TIMEOUT))
        except DashboardError as error:
            snapshot = {"source": "sdk", "cli_error": str(error), "generated_at": generated}
        except ValueError:
            snapshot = {"source": "sdk", "cli_error": "cli_output_not_json", "generated_at": generated}
        else:
            cli_number = str(account.get("account_number", "")) if isinstance(account, dict) else ""
            if expected_account_number and cli_number != str(expected_account_number):
                snapshot = {"source": "sdk", "cli_error": "account_mismatch", "generated_at": generated,
                            "profile": profile}
            else:
                snapshot = {
                    "source": "alpaca-cli",
                    "generated_at": generated,
                    "profile": profile,
                    "clock": clock if isinstance(clock, dict) else {},
                    "account": {k: account.get(k) for k in ACCOUNT_FIELDS if k in account}
                    if isinstance(account, dict) else {},
                    "positions": [
                        {"symbol": p.get("symbol"), "qty": p.get("qty")}
                        for p in positions if isinstance(p, dict)
                    ] if isinstance(positions, list) else [],
                }
    if out_path is not None:
        _write_atomic(out_path, json.dumps(snapshot))
    return snapshot


def expected_account_number(root: Path = ROOT) -> str | None:
    """The engine's account id from the SDK export (logs/account.json), if present."""
    try:
        data = json.loads((root / "logs" / "account.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    number = data.get("account_number") if isinstance(data, dict) else None
    return str(number) if number else None


def export_candles(root: Path = ROOT, days: int = 20) -> str:
    out = candles_dir(root) / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(_py("export_candles.py", "--days", str(days), "--out", str(out)), timeout=CANDLES_TIMEOUT)
    return "ok"


def surge_binary() -> str | None:
    """`surge` on PATH, or the per-user npm install (%APPDATA%\npm) that Anaconda's npm hides."""
    found = shutil.which("surge")
    if found:
        return found
    appdata = os.environ.get("APPDATA", "")
    for name in ("surge.cmd", "surge"):
        candidate = Path(appdata) / "npm" / name
        if appdata and candidate.exists():
            return str(candidate)
    return None


def deploy(root: Path = ROOT) -> str:
    """Push both pages to the surge domains in SURGE_DOMAIN_CYCLES / SURGE_DOMAIN_CANDLES."""
    surge = surge_binary()
    if surge is None:
        return "skipped: surge not installed"
    done = []
    for directory, env_name in ((cycles_dir(root), "SURGE_DOMAIN_CYCLES"), (candles_dir(root), "SURGE_DOMAIN_CANDLES")):
        domain = os.environ.get(env_name, "").strip()
        if not domain:
            continue
        _run([surge, str(directory), domain])
        done.append(domain)
    return "ok: " + ", ".join(done) if done else "skipped: SURGE_DOMAIN_* not set"


def export_all(root: Path = ROOT, *, candles: bool = True, deploy_enabled: bool = True) -> dict[str, str]:
    """Refresh every dashboard file. Returns {step: status}; never raises."""
    profile = os.environ.get("ALPACA_CLI_PROFILE", "").strip() or None
    steps = [
        ("account", lambda: export_account(root)),
        ("pnl", lambda: export_pnl(root)),
        ("config", lambda: write_config_json(root / "settings.yaml", cycles_dir(root) / "config.json")),
        ("journal", lambda: copy_journal(root)),
        ("cli_snapshot", lambda: cli_snapshot(
            profile, cycles_dir(root) / "cli_snapshot.json", expected_account_number(root))),
    ]
    if candles:
        steps.append(("candles", lambda: export_candles(root)))
    if deploy_enabled:
        steps.append(("deploy", lambda: deploy(root)))
    statuses: dict[str, str] = {}
    for name, step in steps:
        try:
            result = step()
            statuses[name] = result if isinstance(result, str) else "ok"
        except Exception as error:  # noqa: BLE001 - a dashboard step must never stop trading
            detail = str(error) if isinstance(error, DashboardError) else type(error).__name__
            statuses[name] = f"failed: {detail}"
            logger.warning("dashboard step {} {}", name, statuses[name])
    return statuses


class _Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # keep the trading log clean
        pass


def serve(port: int = 8080, root: Path = ROOT, *, background: bool = True) -> ThreadingHTTPServer:
    """Serve surge_artifacts/ on 127.0.0.1:<port>; in the background by default."""
    handler = partial(_Handler, directory=str(root / "surge_artifacts"))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    if background:
        threading.Thread(target=server.serve_forever, daemon=True, name="dashboard-http").start()
    else:
        server.serve_forever()
    return server


@app.command("export")
def export_command(candles: bool = typer.Option(True, help="Also refresh the candles page data (slow)."),
                   push: bool = typer.Option(True, help="Deploy to surge when configured.")) -> None:
    """Refresh every dashboard data file (and deploy when surge is configured)."""
    for name, status in export_all(candles=candles, deploy_enabled=push).items():
        typer.echo(f"{name:<12} {status}")


@app.command("serve")
def serve_command(port: int = typer.Option(8080, help="Local port.")) -> None:
    """Serve the dashboard pages from surge_artifacts/ (Ctrl+C to stop)."""
    typer.echo(f"http://localhost:{port}/paca-cycles/   http://localhost:{port}/paca-candles/")
    serve(port, background=False)


@app.command("deploy")
def deploy_command() -> None:
    """Push both pages to the surge domains from the environment."""
    typer.echo(deploy())


if __name__ == "__main__":
    app()
