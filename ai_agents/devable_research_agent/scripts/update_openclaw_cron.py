#!/usr/bin/env python3

"""Update the OpenClaw AI Research Agent cron job message from the prompt file.

OpenClaw's `cron edit --message` breaks on Windows PowerShell when the prompt
contains newlines. This script patches the same jobs.json store directly when
available, or falls back to invoking the OpenClaw CLI via Python subprocess.

Delivery mode must be `none` so the agent posts via the message tool (title +
thread). Set once with:
  openclaw cron edit 811ed28b-8ef8-45b5-a222-cb7d7150f1e0 --no-deliver --tools read,message
"""

import json
import os
import subprocess
import sys
from pathlib import Path

JOB_ID = "811ed28b-8ef8-45b5-a222-cb7d7150f1e0"
PROMPT_FILE = Path(__file__).resolve().parent / "openclaw_digest_prompt.txt"
JOBS_FILE = Path.home() / ".openclaw" / "cron" / "jobs.json"


def _project_root() -> Path:
    env_root = os.environ.get("DEVABLE_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent


def _render_prompt() -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")
    root = str(_project_root()).replace("\\", "/")
    return template.replace("{DEVABLE_PROJECT_ROOT}", root)


def _update_via_jobs_file(prompt: str) -> bool:
    if not JOBS_FILE.exists():
        return False

    data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    updated = False
    for job in data.get("jobs", []):
        if job.get("id") == JOB_ID:
            job.setdefault("payload", {})["message"] = prompt
            updated = True
            break

    if not updated:
        print(f"Error: cron job not found in {JOBS_FILE}: {JOB_ID}", file=sys.stderr)
        return False

    JOBS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated cron job {JOB_ID} in {JOBS_FILE}")
    return True


def _openclaw_node_argv() -> list[str]:
    node = Path(os.environ.get("NODE", r"C:\Program Files\nodejs\node.exe"))
    if not node.exists():
        node = Path("node")
    openclaw_js = (
        Path.home() / "AppData" / "Roaming" / "npm" / "node_modules" / "openclaw" / "dist" / "index.js"
    )
    if openclaw_js.exists():
        return [str(node), str(openclaw_js)]
    return ["openclaw"]


def _update_via_cli(prompt: str) -> bool:
    result = subprocess.run(
        [*_openclaw_node_argv(), "cron", "edit", JOB_ID, "--message", prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return False

    print(f"Updated cron job {JOB_ID} via openclaw CLI")
    return True


def main() -> int:
    if not PROMPT_FILE.exists():
        print(f"Error: prompt file not found: {PROMPT_FILE}", file=sys.stderr)
        return 1

    prompt = _render_prompt()
    if _update_via_jobs_file(prompt) or _update_via_cli(prompt):
        return 0

    print(
        "Error: could not update cron job. Ensure OpenClaw CLI is installed "
        f"or jobs file exists at {JOBS_FILE}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
