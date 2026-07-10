#!/usr/bin/env python3
"""Daily data collection pipeline for the AI Research Agent.

Runs Devable Research Agent collectors, filters top items for digest generation, and logs
output to logs/pipeline.log. Designed to be invoked by Windows Task Scheduler
at 7:45am — OpenClaw reads digest_input_today.json at 8am for the Slack digest.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECT_SCRIPT = PROJECT_ROOT / "skills" / "tracking-list" / "scripts" / "collect.py"
FILTER_SCRIPT = PROJECT_ROOT / "scripts" / "filter_top_items.py"
DATA_FILE = PROJECT_ROOT / "data_today.json"
DIGEST_FILE = PROJECT_ROOT / "digest_input_today.json"
LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"


def _log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)


def _run_step(name: str, cmd: list) -> bool:
    _log(f"START {name}")
    _log(f"CMD {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        _log(f"ERROR {name} failed to launch: {exc}")
        return False

    if result.stdout:
        for line in result.stdout.splitlines():
            _log(f"  {line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            _log(f"  {line}")

    if result.returncode != 0:
        _log(f"ERROR {name} exited with code {result.returncode}")
        return False

    _log(f"DONE {name}")
    return True


def run_pipeline(date_str: str, depth: str = "default", top_n: int = 20) -> int:
    """Run collection and filtering. Returns process exit code."""
    _log("=" * 60)
    _log(f"Pipeline run started (date={date_str}, depth={depth}, top={top_n})")

    if not COLLECT_SCRIPT.exists():
        _log(f"ERROR collect script not found: {COLLECT_SCRIPT}")
        return 1
    if not FILTER_SCRIPT.exists():
        _log(f"ERROR filter script not found: {FILTER_SCRIPT}")
        return 1

    collect_cmd = [
        sys.executable,
        str(COLLECT_SCRIPT),
        "--date", date_str,
        "--depth", depth,
        "--no-cache",
        "--output", str(DATA_FILE),
    ]
    if not _run_step("collect", collect_cmd):
        _log("Pipeline aborted: collection failed")
        return 1

    if not DATA_FILE.exists():
        _log(f"ERROR expected output missing: {DATA_FILE}")
        return 1

    filter_cmd = [
        sys.executable,
        str(FILTER_SCRIPT),
        str(DATA_FILE),
        str(DIGEST_FILE),
        "--top", str(top_n),
    ]
    if not _run_step("filter", filter_cmd):
        _log("Pipeline aborted: filtering failed")
        return 1

    if not DIGEST_FILE.exists():
        _log(f"ERROR expected output missing: {DIGEST_FILE}")
        return 1

    data_kb = DATA_FILE.stat().st_size / 1024
    digest_kb = DIGEST_FILE.stat().st_size / 1024
    _log(f"Pipeline complete: {DATA_FILE.name} ({data_kb:.1f}KB), {DIGEST_FILE.name} ({digest_kb:.1f}KB)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Devable Research Agent daily collection pipeline")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Target collection date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--depth",
        default="default",
        choices=["quick", "default", "deep"],
        help="Collector depth (default: default)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top items for digest input (default: 20)",
    )
    args = parser.parse_args()

    raise SystemExit(run_pipeline(args.date, args.depth, args.top))


if __name__ == "__main__":
    main()
