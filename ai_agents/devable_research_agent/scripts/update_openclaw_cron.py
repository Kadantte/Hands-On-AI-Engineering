#!/usr/bin/env python3
"""Update the OpenClaw AI Research Agent cron job message from the prompt file.

OpenClaw's `cron edit --message` breaks on Windows PowerShell when the prompt
contains newlines. This script patches the same jobs.json store directly.
"""

import json
import sys
from pathlib import Path

JOB_ID = "811ed28b-8ef8-45b5-a222-cb7d7150f1e0"
PROMPT_FILE = Path(__file__).resolve().parent / "openclaw_digest_prompt.txt"
JOBS_FILE = Path.home() / ".openclaw" / "cron" / "jobs.json"


def main() -> int:
    if not PROMPT_FILE.exists():
        print(f"Error: prompt file not found: {PROMPT_FILE}", file=sys.stderr)
        return 1
    if not JOBS_FILE.exists():
        print(f"Error: OpenClaw jobs file not found: {JOBS_FILE}", file=sys.stderr)
        return 1

    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))

    updated = False
    for job in data.get("jobs", []):
        if job.get("id") == JOB_ID:
            job.setdefault("payload", {})["message"] = prompt
            updated = True
            break

    if not updated:
        print(f"Error: cron job not found: {JOB_ID}", file=sys.stderr)
        return 1

    JOBS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated cron job {JOB_ID} in {JOBS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
