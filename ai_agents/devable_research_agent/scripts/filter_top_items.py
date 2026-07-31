#!/usr/bin/env python3
"""
Filter top scored items from the daily collection JSON.
Outputs a smaller JSON file with only the top N items for digest generation.

Applies category diversity quotas so the digest is not dominated by a single source.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from mistralai.client import Mistral

    _HAS_MISTRAL = True
except ImportError:
    _HAS_MISTRAL = False

MORNING_AI_ENV_FILE = Path.home() / ".config" / "morning-ai" / ".env"
PROJECT_ROOT = Path(
    os.environ.get("DEVABLE_PROJECT_ROOT", "")
).resolve() if os.environ.get("DEVABLE_PROJECT_ROOT") else Path(__file__).resolve().parent.parent
DELIVERED_ITEMS_PATH = PROJECT_ROOT / "logs" / "delivered_items.json"
PIPELINE_LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"
PAPER_SCORES_PATH = PROJECT_ROOT / "logs" / "paper_scores_today.json"
PAPER_SCORE_CACHE_PATH = PROJECT_ROOT / "logs" / "paper_score_cache.json"
REJECTED_ITEMS_PATH = PROJECT_ROOT / "logs" / "rejected_items_today.json"
FILTER_DEBUG_TRACE_PATH = PROJECT_ROOT / "logs" / "filter_debug_trace.txt"
PRIORITY_REPOS_PATH = PROJECT_ROOT / "entities" / "custom" / "priority-repos.md"
DEFAULT_DELIVERED_LOOKBACK_DAYS = 7
GITHUB_REPO_DELIVERED_LOOKBACK_DAYS = 14
GITHUB_TRENDING_DELIVERED_LOOKBACK_DAYS = 3
AINEWS_DELIVERED_LOOKBACK_DAYS = 3
HN_SHOW_LAUNCH_DELIVERED_LOOKBACK_DAYS = 3
REPO_FREQUENCY_CAP = 2
REPO_FREQUENCY_LOOKBACK_DAYS = 30
VERIFY_MEMORY_LOOKBACK_DAYS = 7
TZ_CN = timezone(timedelta(hours=8))

# Per-source date cutoffs for stale-item filtering
SOURCE_DATE_CUTOFF_HOURS = {
    "arxiv": 72,
    "github": 48,
    "web": 48,
    "huggingface": 72,
    "x": 72,
}
SOURCE_DATE_CUTOFF_DAYS = {
    "repo": 7,
}
DEFAULT_DATE_CUTOFF_HOURS = 48

PAPER_LLM_MODEL = "mistral-small-latest"
VERIFY_LLM_MODEL = "mistral-small-latest"
DESCRIPTION_LLM_MODEL = "mistral-small-latest"
MIN_PAPER_LLM_SCORE = 4
MIN_PAPER_HYBRID_IMPORTANCE = 4.5
FAILED_PAPER_LLM_SCORE = 3
PAPER_LLM_MAX_WORKERS = 8

_PAPER_LLM_SYSTEM = (
    "You are an editor for an AI engineering newsletter for developers who "
    "build with LLMs and agents. Score paper relevance strictly."
)

_PAPER_LLM_USER_TEMPLATE = """Score this paper's relevance to AI engineers who build, train, or deploy AI systems. Reply with ONLY a single digit 1-5.

5 = core AI engineering: LLMs, agents, reasoning, inference, fine-tuning, benchmarks, RAG, tool use, multimodal models
4 = strong engineering value: new techniques builders can use today
3 = tangentially related: uses AI but not about building AI systems
2 = domain application of AI: medical, legal, social, scientific
1 = not relevant: math, music, networking, other domains

Title: {title}
Summary: {summary}"""

_VERIFY_LLM_SYSTEM = (
    "You are an editorial verifier for an AI engineering digest. "
    "Return only a JSON array — no prose, no markdown fences."
)

_VERIFY_LLM_USER_TEMPLATE = """You are an editorial verifier for an AI engineering digest.

Here are the items selected for today's digest:
{top_items_json}

Here are items delivered in the last 14 days (memory spine):
{memory_spine_json}

Your job: remove any item from today's list ONLY when:
1. It has the exact same source_url as an item in the memory spine, OR
2. It has the same entity AND same content_type as an item in the memory spine

Do NOT remove AINews items (source_label: AINews) unless the source_url is an exact duplicate.
Do NOT remove items for similar topics, paraphrased titles, or related stories.

Return only the filtered list as a JSON array with the same structure."""

_DESCRIPTION_LLM_SYSTEM = (
    "You write one-line technical descriptions for an AI engineering digest. "
    "Reply with ONLY the description text — no quotes, no preamble."
)

_DESCRIPTION_LLM_USER_TEMPLATE = """Write ONE technical sentence (max 12 words) describing what changed and why it matters to AI engineers who build with LLMs and agents.

Title: {title}
Summary: {summary}
Source: {source}

Reply with only the description."""

_PRIORITY_REPOS_CACHE: Set[str] = set()

# Per-category min/max quotas for digest diversity
QUOTA_LIMITS = {
    "paper": {"min": 2, "max": 4},
    "github": {"min": 2, "max": 8},
    "web": {"min": 1, "max": 3},
    "ainews": {"min": 2, "max": 4},
    "x": {"min": 1, "max": 4},
    "hackernews": {"min": 1, "max": 4},
    "huggingface": {"min": 0, "max": 2},
    "good_ai_list": {"min": 3, "max": 6},
}

GOOD_AI_LIST_IMPORTANCE_BOOST = 0.5
# Previously-delivered Good AI List repos may not fill quota slots (fresh repos only).
GOOD_AI_LIST_MAX_REDELIVERED = 0

_paper_llm_scores: Dict[str, int] = {}
_MISTRAL_CLIENT: "Mistral | None" = None

_HF_INCLUDE_KEYWORDS = [
    "llm",
    "language model",
    "agent",
    "reasoning",
    "inference",
    "embedding",
    "multimodal",
    "code",
    "text generation",
    "vision",
    "audio",
]

_HF_SKIP_KEYWORDS = [
    "molecular docking",
    "docking",
    "diffdock",
    "chemistry",
    "biology",
    "medical imaging",
    "protein",
    "molecule",
    "chemoinformatics",
]

_GITHUB_MAINTENANCE_KEYWORDS = [
    "migrate ci",
    "bump",
    "chore",
    "relicense",
    "dependency update",
    "fix typo",
    "shell expansion",
    "removed shell",
    "remove shell",
    "minor fix",
    "cleanup",
    "refactor",
    "remove unused",
    "update readme",
    "documentation",
    "updated bundled",
    "bundled claude cli",
    "updated to parity",
    "fixed hook events",
]

_GITHUB_FEATURE_KEYWORDS = [
    "new feature",
    "new capability",
    "capability",
    "capabilities",
    "improvement",
    "improve",
    "introduce",
    "introduces",
    "adds support",
    "add support",
    "now supports",
    "enable",
    "enables",
    "performance",
    "faster",
    "benchmark",
    "launch",
    "announce",
    "new api",
    "breaking change",
]

# Claude Agent SDK parity releases — wiring/stream-json sync, not product news.
_SDK_SYNC_KEYWORDS = [
    "updated bundled",
    "bundled claude cli",
    "updated to parity",
    "command_lifecycle",
    "duration_api_ms",
    "stream-json",
    "timedoutafterms",
    "bashtooloutput",
    "auto-backgrounded",
]

# Skip fix-heavy patch releases only when Added/Improved bullets lack major capability.
_MAJOR_RELEASE_CAPABILITY_KEYWORDS = [
    "new api",
    "breaking change",
    "launch",
    "benchmark",
    "agent framework",
    "mcp server",
    "multi-agent",
    "openapi",
]

# GitHub Trending only — finance/trading repos are not AI engineering news.
_GITHUB_TRENDING_FINANCE_KEYWORDS = [
    "stock",
    "trading",
    "forex",
    "crypto trading",
    "investment",
]

# GitHub Trending only — Devable Research Agent ecosystem plugin repos should not surface.
_GITHUB_TRENDING_PLUGIN_KEYWORDS = [
    "devable-research-agent skill",
    "last30days",
    "tracking-list skill",
]

# Thin commercial-AI plugin wrappers (not novel engineering).
_GITHUB_LOW_SIGNAL_PLUGIN_KEYWORDS = [
    "codex-plugin",
    "plugin-cc",
    "claude-plugin",
    "claude code plugin",
    "use codex from claude",
]

# GitHub Trending — must signal AI engineering in name or description.
_GITHUB_TRENDING_AI_KEYWORDS = [
    "llm",
    "ai",
    "agent",
    "model",
    "inference",
    "embedding",
    "rag",
    "fine-tune",
    "finetune",
    "transformer",
    "diffusion",
    "neural",
    "gpt",
    "claude",
    "llama",
    "mistral",
    "deepseek",
    "qwen",
    "benchmark",
    "eval",
    "prompt",
    "skill",
    "mcp",
    "codex",
    "open source ai",
    "machine learning",
    "deep learning",
]

# GitHub Trending only — offensive-security cheat-sheet launchers, not AI engineering.
_GITHUB_TRENDING_CYBERSECURITY_PHRASES = [
    "cybersecurity cheat-sheet",
    "penetration testing",
    "just install and start hacking",
]

# Explicit owner/repo blocklist (GitHub Trending + Good AI List).
_REPO_BLOCKLIST = {
    "alibaba/page-agent",
    "jcodesmore/ai-website-cloner-template",
    "xbtlin/ai-berkshire",
    "synthetic-sciences/openscience",
    "heygen-com/hyperframes",
    "isjiamu/gzh-design-skill",
    "mdx-tom/gpt-5.6-instruct",
    "zhishile/codex-auth-helper",
    "jia-ethan/codex-keysmith",
}

# Jailbreak packs, auth/session exporters — not AI engineering tooling.
_REPO_NON_ENGINEERING_KEYWORDS = [
    "jailbreak",
    "破甲",
    "auth.json",
    "session export",
    "export your logged-in",
    "codex登陆",
    "auth helper",
    "codex-keysmith",
    "codex keysmith",
    "破甲提示词",
]

# Skip repos whose primary purpose is cloning, generic scraping, or standalone browser automation.
_REPO_WEBSITE_CLONER_KEYWORDS = [
    "website cloner",
    "clone any website",
    "site cloner",
    "website-cloner",
    "ai-website-cloner",
    "clone website",
    "replicate website",
    "website replication",
    "website cloner template",
]

_REPO_GENERIC_SCRAPING_KEYWORDS = [
    "web scraper",
    "web scraping",
    "scrape websites",
    "scraping tool",
    "scrape the web",
    "website crawler",
    "crawl the web",
    "crawl websites",
]

_REPO_BROWSER_AUTOMATION_KEYWORDS = [
    "browser automation",
    "automate your browser",
    "headless browser",
    "browser control",
    "browser agent",
    "puppeteer",
    "playwright automation",
    "selenium automation",
]

# Scraping/automation is OK when clearly part of an AI agent framework.
_REPO_AI_AGENT_FRAMEWORK_SIGNALS = [
    "ai agent",
    "coding agent",
    "agent framework",
    "agent workflow",
    "llm",
    "language model",
    "mcp",
    "rag",
    "agent skill",
    "agent tool",
    "for agents",
    "agent-ready",
]

_WEB_SKIP_KEYWORDS = [
    "legal",
    "law firm",
    "harvey",
    "compliance",
    "healthcare",
    "climate",
    "environment",
    "drone",
    "satellite",
    "agriculture",
    "music",
    "suno",
    "raises $",
    "funding round",
    "series a",
    "series b",
    "valuation",
    "techcrunch",
    "investors",
    "venture capital",
]

_WEB_RELEASE_KEYWORDS = [
    "release",
    "released",
    "launch",
    "launched",
    "announce",
    "announcing",
    "introducing",
    "open-weight",
    "open weight",
    "model weights",
    "weights available",
    "now available",
    "new version",
    "open source",
    "open-source",
]

_WEB_NEWS_DOMAINS = [
    "techcrunch.com",
    "venturebeat.com",
    "theverge.com",
    "wired.com",
    "bloomberg.com",
    "reuters.com",
    "cnbc.com",
]

_WEB_RELEVANCE_KEYWORDS = [
    "llm",
    "large language model",
    "language model",
    "model release",
    "open-weight",
    "open source",
    "open-source",
    "agent",
    "framework",
    "benchmark",
    "leaderboard",
    "inference",
    "fine-tun",
    "finetun",
    "transformer",
    "multimodal",
    "embedding",
    "vllm",
    "sglang",
    "rag",
    "tool use",
    "coding agent",
    "github",
    "huggingface",
    "weights",
    "dataset",
    "sdk",
    "api",
    "release",
    "launch",
    "announc",
    "reasoning",
    "diffusion",
    "neural",
    "machine learning",
    "deep learning",
    "ai engineering",
    "build agent",
    "developer tool",
    "context window",
    "coding",
    "openai",
    "anthropic",
    "gemini",
    "claude",
    "mistral",
    "llama",
    "qwen",
    "deepseek",
]

_HN_INCLUDE_KEYWORDS = [
    "open source",
    "open-source",
    "llm",
    "agent",
    "inference",
    "benchmark",
    "framework",
    "fine-tuning",
    "rag",
    "embedding",
    "model weights",
    "arxiv",
    "paper",
    "tool",
    "library",
    "repo",
    "github",
    "release",
    "local model",
    "self-hosted",
    "vllm",
    "llama",
    "mistral",
    "deepseek",
    "qwen",
    "flux",
    "diffusion",
    "transformer",
    "training",
    "weights",
    "distillation",
    "quantization",
    "mcp",
    "skill",
    "evaluation",
    "evals",
    "api",
    "sdk",
    "coding",
    "model",
]

_HN_ENGINEERING_TITLE_TERMS = [
    "agent",
    "api",
    "sdk",
    "coding",
    "release",
    "benchmark",
    "model",
]

_HN_CLOSED_COMMERCIAL = [
    "gpt",
    "claude",
    "gemini",
    "chatgpt",
    "copilot",
    "sora",
]

_HN_OPEN_SIGNAL = [
    "open source",
    "open-source",
    "open weight",
    "open-weight",
    "open weights",
]


_GITHUB_REPO_RE = re.compile(
    r"github\.com/([^/?#]+)/([^/?#]+)",
    re.IGNORECASE,
)
_GITHUB_SLUG_RE = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")

_X_QUALITY_KEYWORDS = [
    "model",
    "agent",
    "llm",
    "open source",
    "release",
    "benchmark",
    "inference",
    "tool",
    "framework",
    "paper",
    "arxiv",
    "weights",
    "fine-tune",
    "fine tune",
    "dataset",
    "api",
    "sdk",
    "github",
    "repo",
    "training",
    "reasoning",
    "embedding",
    "rag",
    "mcp",
    "coding",
    "engineering",
]
_X_MIN_VIEWS = 200
_X_MIN_LIKES = 5


def _github_release_tag(item: Dict) -> str:
    """Extract semver/tag from a GitHub release URL or title."""
    url = item.get("source_url", "") or ""
    match = re.search(r"/releases/tag/([^/?#]+)", url, re.IGNORECASE)
    if match:
        return match.group(1).lower().lstrip("v")
    title = (item.get("title") or "").strip()
    match = re.search(r"@([\d.]+(?:[-\w.+]+)?)\s*$", title)
    if match:
        return match.group(1).lower()
    match = re.search(r"\s(v?\d+\.\d+(?:\.\d+)?(?:[-\w.+]+)?)\s*$", title, re.IGNORECASE)
    if match:
        return match.group(1).lower().lstrip("v")
    return ""


def _is_github_release(item: Dict) -> bool:
    return item.get("source") == "github" and not _is_github_trending(item)


def _is_hn_show_or_launch(item: Dict) -> bool:
    if item.get("source") != "hackernews":
        return False
    title = (item.get("title") or "").strip().lower()
    return title.startswith("show hn:") or title.startswith("launch hn:")


def _delivered_dedup_key(item: Dict) -> str:
    """Canonical dedup key for delivered-log matching."""
    if _is_good_ai_list(item) or item.get("source") == "arxiv":
        return _normalize_source_url(item.get("source_url", ""))

    if _is_github_trending(item):
        repo_key = _github_repo_key(item)
        if repo_key:
            return f"github-trending:{repo_key}"
        return _normalize_source_url(item.get("source_url", ""))

    if _is_github_release(item):
        repo_key = _github_repo_key(item)
        tag = _github_release_tag(item)
        if repo_key and tag:
            return f"github-release:{repo_key}@{tag}"
        if repo_key:
            return f"https://github.com/{repo_key}"

    normalized = _normalize_source_url(item.get("source_url", ""))
    if normalized:
        return normalized
    return (item.get("source_url") or "").split("?")[0].rstrip("/").lower()


def _github_repo_key(item_or_url: Any) -> str:
    """Return canonical owner/repo slug for GitHub dedup, or empty string."""
    if isinstance(item_or_url, dict):
        stored = (item_or_url.get("repo_key") or "").strip().lower()
        if stored and "/" in stored and " " not in stored:
            return stored
        source = item_or_url.get("source", "")
        url = item_or_url.get("source_url", "")
        title = item_or_url.get("title", "")
        if source not in ("github", "repo") and "github.com/" not in url.lower():
            slug_match = _GITHUB_SLUG_RE.match((title or "").strip())
            if slug_match:
                return f"{slug_match.group(1).lower()}/{slug_match.group(2).lower()}"
            return ""
        for text in (url, title):
            match = _GITHUB_REPO_RE.search(text or "")
            if match:
                return f"{match.group(1).lower()}/{match.group(2).lower()}"
            slug_match = _GITHUB_SLUG_RE.match((text or "").strip())
            if slug_match:
                return f"{slug_match.group(1).lower()}/{slug_match.group(2).lower()}"
        return ""
    match = _GITHUB_REPO_RE.search(str(item_or_url or ""))
    if match:
        return f"{match.group(1).lower()}/{match.group(2).lower()}"
    slug_match = _GITHUB_SLUG_RE.match(str(item_or_url or "").strip())
    if slug_match:
        return f"{slug_match.group(1).lower()}/{slug_match.group(2).lower()}"
    return ""


def _normalize_source_url(url: str) -> str:
    """Normalize source_url for delivered-item deduplication."""
    if not url:
        return ""
    clean = url.split("?")[0].rstrip("/").lower()
    repo_key = _github_repo_key(clean)
    if repo_key:
        return f"https://github.com/{repo_key}"
    return clean


def _parse_item_date(date_value: Optional[str]) -> Optional[datetime]:
    """Parse an item date string; return None if missing or unparseable."""
    if not date_value:
        return None
    text = str(date_value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(text[:10], fmt)
            return parsed.replace(tzinfo=TZ_CN)
        except ValueError:
            continue
    return None


def _item_date_cutoff(item: Dict, now: datetime) -> datetime:
    """Return the oldest acceptable datetime for an item's source."""
    source = item.get("source", "")
    days = SOURCE_DATE_CUTOFF_DAYS.get(source)
    if days is not None:
        return now - timedelta(days=days)
    hours = SOURCE_DATE_CUTOFF_HOURS.get(source, DEFAULT_DATE_CUTOFF_HOURS)
    return now - timedelta(hours=hours)


def filter_stale_items(items: List[Dict]) -> Tuple[List[Dict], int, Dict[str, int]]:
    """Drop items older than their source-specific cutoff. Items without a date are kept."""
    now = datetime.now(TZ_CN)
    fresh: List[Dict] = []
    skipped = 0
    by_source: Dict[str, int] = {}

    for item in items:
        parsed = _parse_item_date(item.get("date"))
        source = item.get("source") or "unknown"
        if source == "x":
            confidence = str(item.get("date_confidence", "")).strip().lower()
            if not item.get("date") or confidence == "low" or parsed is None:
                fresh.append(item)
                by_source[source] = by_source.get(source, 0) + 1
                continue
        if parsed is None:
            fresh.append(item)
            by_source[source] = by_source.get(source, 0) + 1
            continue
        if parsed < _item_date_cutoff(item, now):
            skipped += 1
            continue
        fresh.append(item)
        by_source[source] = by_source.get(source, 0) + 1

    return fresh, skipped, by_source


def _delivered_lookback_days(item: Dict) -> int:
    """URL dedup window: 14d GAL/releases, 3d Trending/AINews/Show HN, 7d default."""
    if _is_good_ai_list(item):
        return GITHUB_REPO_DELIVERED_LOOKBACK_DAYS
    if _is_github_trending(item):
        return GITHUB_TRENDING_DELIVERED_LOOKBACK_DAYS
    if _is_ainews(item):
        return AINEWS_DELIVERED_LOOKBACK_DAYS
    if _is_hn_show_or_launch(item):
        return HN_SHOW_LAUNCH_DELIVERED_LOOKBACK_DAYS
    source = item.get("source", "")
    if source in ("github", "repo"):
        return GITHUB_REPO_DELIVERED_LOOKBACK_DAYS
    return DEFAULT_DELIVERED_LOOKBACK_DAYS


def _pipeline_log(message: str) -> None:
    PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S")
    with PIPELINE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _load_priority_repos(path: Path = PRIORITY_REPOS_PATH) -> Set[str]:
    """Load owner/repo slugs exempt from the frequency cap."""
    global _PRIORITY_REPOS_CACHE
    if _PRIORITY_REPOS_CACHE:
        return _PRIORITY_REPOS_CACHE
    repos: Set[str] = set()
    if not path.exists():
        return repos
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _GITHUB_REPO_RE.search(line)
        if match:
            repos.add(f"{match.group(1).lower()}/{match.group(2).lower()}")
    _PRIORITY_REPOS_CACHE = repos
    return repos


def _is_priority_repo(item: Dict) -> bool:
    repo_key = _github_repo_key(item)
    return bool(repo_key and repo_key in _load_priority_repos())


def _topic_summary(title: str, summary: str = "") -> str:
    """Derive a short 3-word topic label for the memory spine."""
    stop = {
        "the", "a", "an", "for", "on", "and", "or", "to", "with", "of", "in",
        "is", "are", "was", "new", "from", "by", "at", "as", "its", "via",
    }
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+-]*", f"{title} {summary}"[:160])
    picked = [w for w in words if w.lower() not in stop][:3]
    if len(picked) >= 2:
        return " ".join(picked)
    return " ".join((title or "unknown").split()[:3])


def _load_delivered_store(path: Path = DELIVERED_ITEMS_PATH) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load delivery log entries and per-repo frequency metadata."""
    if not path.exists():
        return [], {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return [], {}

    if isinstance(data, list):
        entries = [e for e in data if isinstance(e, dict)]
        return entries, _derive_frequency_from_entries(entries)

    entries = [e for e in data.get("deliveries", []) if isinstance(e, dict)]
    frequency = data.get("frequency", {})
    if not isinstance(frequency, dict):
        frequency = {}
    frequency = {
        str(repo_key).lower(): record
        for repo_key, record in frequency.items()
        if isinstance(record, dict)
    }
    if not frequency and entries:
        frequency = _derive_frequency_from_entries(entries)
    return entries, frequency


def _load_delivered_entries(path: Path = DELIVERED_ITEMS_PATH) -> List[Dict]:
    """Load all delivered-item log entries."""
    entries, _ = _load_delivered_store(path)
    return entries


def _frequency_key_for_item(item: Dict) -> Optional[str]:
    """Return repo key for frequency tracking (GitHub + Good AI List repos only)."""
    source = item.get("source", "")
    if source not in ("github", "repo"):
        return None
    repo_key = _github_repo_key(item)
    return repo_key or None


def _derive_frequency_from_entries(entries: List[Dict]) -> Dict[str, Dict]:
    """Build frequency map by counting repo deliveries in the lookback window."""
    cutoff = (
        datetime.now(TZ_CN) - timedelta(days=REPO_FREQUENCY_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    counts: Dict[str, Dict] = {}
    for entry in entries:
        if entry.get("date", "") < cutoff:
            continue
        repo_key = _github_repo_key(entry)
        if not repo_key:
            stored = (entry.get("repo_key") or "").strip().lower()
            if stored and "/" in stored:
                repo_key = stored
        if not repo_key:
            continue
        record = counts.setdefault(
            repo_key,
            {
                "delivery_count": 0,
                "last_delivered": entry.get("date", ""),
                "source_url": _normalize_source_url(entry.get("source_url", "")),
                "title": entry.get("title", ""),
            },
        )
        record["delivery_count"] = int(record.get("delivery_count", 0)) + 1
        if entry.get("date", "") >= record.get("last_delivered", ""):
            record["last_delivered"] = entry.get("date", "")
            if entry.get("source_url"):
                record["source_url"] = _normalize_source_url(entry.get("source_url", ""))
            if entry.get("title"):
                record["title"] = entry.get("title", "")
    return counts


def _load_repo_delivery_counts(path: Path = DELIVERED_ITEMS_PATH) -> Dict[str, int]:
    """Return delivery counts per repo key within the frequency lookback window."""
    entries, frequency = _load_delivered_store(path)
    derived = _derive_frequency_from_entries(entries)
    counts: Dict[str, int] = {}
    for repo_key, record in derived.items():
        counts[repo_key] = int(record.get("delivery_count", 0))
    for repo_key, record in frequency.items():
        if not isinstance(record, dict):
            continue
        counts[repo_key] = max(counts.get(repo_key, 0), int(record.get("delivery_count", 0)))
    return counts


def _repo_delivery_count(item: Dict) -> int:
    """How many times this repo was delivered in the frequency lookback window."""
    repo_key = _frequency_key_for_item(item)
    if not repo_key:
        return 0
    return _load_repo_delivery_counts().get(repo_key, 0)


def _is_frequency_capped(item: Dict) -> bool:
    """Block repos delivered REPO_FREQUENCY_CAP+ times in the last 30 days."""
    if _is_priority_repo(item):
        return False
    repo_key = _frequency_key_for_item(item)
    if not repo_key:
        return False
    return _repo_delivery_count(item) >= REPO_FREQUENCY_CAP


def filter_frequency_capped(items: List[Dict]) -> Tuple[List[Dict], int, List[str]]:
    """Remove repos that hit the 30-day frequency cap."""
    kept: List[Dict] = []
    skipped = 0
    capped_titles: List[str] = []
    counts = _load_repo_delivery_counts()
    for item in items:
        if _is_priority_repo(item):
            kept.append(item)
            continue
        repo_key = _frequency_key_for_item(item)
        if repo_key and counts.get(repo_key, 0) >= REPO_FREQUENCY_CAP:
            skipped += 1
            capped_titles.append(item.get("title") or repo_key)
            continue
        kept.append(item)
    return kept, skipped, capped_titles


def _load_recent_delivered_urls(path: Path = DELIVERED_ITEMS_PATH) -> set:
    """Return source_urls delivered within each item-type lookback window."""
    entries = _load_delivered_entries(path)
    if not entries:
        return set()

    now = datetime.now(TZ_CN)
    urls: set = set()
    for entry in entries:
        normalized = _normalize_source_url(entry.get("source_url", ""))
        if not normalized:
            continue
        entry_date = entry.get("date", "")
        if not entry_date:
            continue
        # Use the longest lookback so we do not over-filter; per-item check in filter_already_delivered
        cutoff = (now - timedelta(days=GITHUB_REPO_DELIVERED_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        if entry_date >= cutoff:
            urls.add(normalized)
            repo_key = _github_repo_key(entry)
            if repo_key:
                urls.add(f"https://github.com/{repo_key}")
    return urls


def _load_memory_spine(lookback_days: int = VERIFY_MEMORY_LOOKBACK_DAYS) -> List[Dict]:
    """Return enriched delivered items from the last N days for LLM verification."""
    cutoff_date = (datetime.now(TZ_CN) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    spine: List[Dict] = []
    for entry in _load_delivered_entries():
        if entry.get("date", "") < cutoff_date:
            continue
        spine.append({
            "source_url": entry.get("source_url", ""),
            "date": entry.get("date", ""),
            "title": entry.get("title", ""),
            "entity": entry.get("entity", ""),
            "content_type": entry.get("content_type", ""),
            "topic": entry.get("topic", ""),
        })
    return spine


def _count_delivered_candidate_matches(items: List[Dict]) -> Tuple[int, int, int]:
    """Count how many candidates match the delivered log (by URL or GitHub repo key)."""
    entries = _load_delivered_entries()
    if not entries:
        return 0, 0, 0

    now = datetime.now(TZ_CN)
    recent_urls: set = set()
    recent_repo_keys: set = set()
    for entry in entries:
        entry_date = entry.get("date", "")
        if not entry_date:
            continue
        cutoff = (now - timedelta(days=GITHUB_REPO_DELIVERED_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        if entry_date < cutoff:
            continue
        normalized = _normalize_source_url(entry.get("source_url", ""))
        if normalized:
            recent_urls.add(normalized)
        repo_key = _github_repo_key(entry)
        if repo_key:
            recent_repo_keys.add(repo_key)

    url_matches = 0
    repo_matches = 0
    for item in items:
        normalized = _normalize_source_url(item.get("source_url", ""))
        repo_key = _github_repo_key(item)
        if normalized and normalized in recent_urls:
            url_matches += 1
        elif repo_key and repo_key in recent_repo_keys:
            repo_matches += 1
    return url_matches + repo_matches, url_matches, repo_matches


def _recent_delivery_entry(item: Dict) -> Optional[Dict]:
    """Return the delivered-log entry for an item within its lookback window, if any."""
    entries = _load_delivered_entries()
    if not entries:
        return None

    now = datetime.now(TZ_CN)
    entry_by_key: Dict[str, Dict] = {}
    entry_by_repo_key: Dict[str, Dict] = {}
    for entry in entries:
        key = _delivered_dedup_key(entry)
        if key:
            entry_by_key[key] = entry
        repo_key = _github_repo_key(entry)
        if repo_key:
            entry_by_repo_key[repo_key] = entry

    dedup_key = _delivered_dedup_key(item)
    entry = entry_by_key.get(dedup_key) if dedup_key else None
    if entry is None and _is_github_trending(item):
        repo_key = _github_repo_key(item)
        if repo_key:
            entry = entry_by_repo_key.get(repo_key)

    if not entry:
        return None

    lookback = _delivered_lookback_days(item)
    cutoff = (now - timedelta(days=lookback)).strftime("%Y-%m-%d")
    if entry.get("date", "") >= cutoff:
        return entry
    return None


def _was_recently_delivered(item: Dict) -> bool:
    """True if this item appears in delivered_items.json within its lookback window."""
    return _recent_delivery_entry(item) is not None


def filter_already_delivered(items: List[Dict], delivered_urls: set) -> Tuple[List[Dict], int]:
    """Skip items whose dedup key was already posted within the lookback window.

    Good AI List repos use the same 14-day block as GitHub releases (no redelivery).
    arXiv papers are kept so the paper minimum quota can be met when fresh papers are scarce.
    GitHub releases dedup by repo+semver; GitHub Trending dedups by bare repo slug.
    """
    entries = _load_delivered_entries()
    if not entries and not delivered_urls:
        return items, 0

    now = datetime.now(TZ_CN)
    entry_by_key: Dict[str, Dict] = {}
    entry_by_repo_key: Dict[str, Dict] = {}
    for entry in entries:
        key = _delivered_dedup_key(entry)
        if key:
            entry_by_key[key] = entry
        repo_key = _github_repo_key(entry)
        if repo_key:
            entry_by_repo_key[repo_key] = entry

    kept: List[Dict] = []
    skipped = 0
    for item in items:
        if item.get("source") == "arxiv":
            kept.append(item)
            continue

        dedup_key = _delivered_dedup_key(item)
        entry = entry_by_key.get(dedup_key) if dedup_key else None
        if entry is None and _is_github_trending(item):
            repo_key = _github_repo_key(item)
            if repo_key:
                entry = entry_by_repo_key.get(repo_key)

        if entry:
            lookback = _delivered_lookback_days(item)
            cutoff = (now - timedelta(days=lookback)).strftime("%Y-%m-%d")
            if entry.get("date", "") >= cutoff:
                skipped += 1
                continue
        elif _is_github_release(item):
            pass
        elif dedup_key and dedup_key in delivered_urls:
            skipped += 1
            continue
        elif _is_github_trending(item):
            repo_key = _github_repo_key(item)
            bare_repo = f"https://github.com/{repo_key}" if repo_key else ""
            if bare_repo and bare_repo in delivered_urls:
                skipped += 1
                continue
        kept.append(item)
    return kept, skipped


def _append_delivered_items(
    items: List[Dict],
    delivery_date: str,
    path: Path = DELIVERED_ITEMS_PATH,
) -> None:
    """Append selected digest items and increment per-repo delivery counts."""
    entries, frequency = _load_delivered_store(path)
    cutoff = (
        datetime.now(TZ_CN) - timedelta(days=REPO_FREQUENCY_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")

    for item in items:
        source_url = (item.get("source_url") or "").strip()
        if not source_url:
            continue
        title = (item.get("title") or "").strip()
        normalized_url = _normalize_source_url(source_url) or source_url
        repo_key = _frequency_key_for_item(item)

        prev_count = 0
        if repo_key:
            derived = _derive_frequency_from_entries(entries)
            prev_count = max(
                int(frequency.get(repo_key, {}).get("delivery_count", 0)),
                int(derived.get(repo_key, {}).get("delivery_count", 0)),
            )
        new_count = prev_count + 1 if repo_key else 1

        entries.append({
            "source_url": normalized_url,
            "date": delivery_date,
            "title": title,
            "entity": (item.get("entity") or "").strip(),
            "content_type": (item.get("content_type") or "").strip(),
            "topic": _topic_summary(title, item.get("summary") or ""),
            "delivery_count": new_count,
            "repo_key": repo_key or "",
        })

        if repo_key:
            frequency[repo_key] = {
                "delivery_count": new_count,
                "last_delivered": delivery_date,
                "source_url": normalized_url,
                "title": title,
            }

    # Drop frequency records older than the lookback window (counts reset after 30d).
    pruned_frequency: Dict[str, Dict] = {}
    for repo_key, record in frequency.items():
        if not isinstance(record, dict):
            continue
        if record.get("last_delivered", "") >= cutoff:
            pruned_frequency[repo_key] = record

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"deliveries": entries, "frequency": pruned_frequency}, handle, ensure_ascii=False, indent=2)


def _item_key(item: Dict) -> str:
    return item.get("id") or item.get("source_url") or item.get("title", "")


def _is_good_ai_list(item: Dict) -> bool:
    return item.get("source_label") == "Good AI List" or (
        item.get("source") == "repo" and item.get("source_label") == "Good AI List"
    )


def _is_ainews(item: Dict) -> bool:
    return item.get("source_label") == "AINews" or str(item.get("entity", "")).startswith("AINews")


def _is_ainews_x(item: Dict) -> bool:
    """AINews story with a direct X/Twitter URL (digest → What's Trending on X)."""
    if not _is_ainews(item):
        return False
    url = (item.get("source_url") or "").lower()
    return "x.com" in url or "twitter.com" in url


def _load_morning_ai_env() -> None:
    """Load Devable Research Agent .env files into os.environ without overriding existing vars."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from lib.env import load_env_file

    merged: Dict[str, str] = {}
    for path in (
        MORNING_AI_ENV_FILE,
        project_root / ".claude" / "morning-ai.env",
        project_root / ".env",
        project_root / ".env.local",
    ):
        merged.update(load_env_file(str(path)))

    for key, value in merged.items():
        if value and key not in os.environ:
            os.environ[key] = value


def _load_mistral_api_key() -> Optional[str]:
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    return key or None


def _mistral_client(api_key: str) -> "Mistral":
    global _MISTRAL_CLIENT
    if _MISTRAL_CLIENT is None:
        _MISTRAL_CLIENT = Mistral(api_key=api_key)
    return _MISTRAL_CLIENT


def _paper_cache_key(item: Dict) -> str:
    title = item.get("title", "") or ""
    summary = (item.get("summary") or "")[:300]
    return f"{title}|{summary}"


_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([0-9]+\.[0-9]+)(?:v\d+)?", re.IGNORECASE)


def _arxiv_id(item: Dict) -> str:
    """Return canonical arXiv ID (without version suffix) for cache lookup."""
    url = item.get("source_url", "") or ""
    match = _ARXIV_ID_RE.search(url)
    if match:
        return match.group(1)
    item_id = item.get("id", "") or ""
    if item_id.startswith("ARXIV-"):
        return item_id.replace("ARXIV-", "").split("v")[0]
    return ""


def _paper_prompt_version_hash() -> str:
    payload = _PAPER_LLM_SYSTEM + _PAPER_LLM_USER_TEMPLATE + PAPER_LLM_MODEL
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_paper_score_cache(prompt_hash: str) -> Dict[str, int]:
    if not PAPER_SCORE_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(PAPER_SCORE_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if data.get("prompt_hash") != prompt_hash:
        return {}
    entries = data.get("entries", {})
    scores: Dict[str, int] = {}
    for arxiv_id, record in entries.items():
        if isinstance(record, dict):
            scores[arxiv_id] = int(record.get("score", FAILED_PAPER_LLM_SCORE))
        else:
            scores[arxiv_id] = int(record)
    return scores


def _save_paper_score_cache(prompt_hash: str, scores: Dict[str, int]) -> None:
    PAPER_SCORE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Dict] = {}
    if PAPER_SCORE_CACHE_PATH.exists():
        try:
            data = json.loads(PAPER_SCORE_CACHE_PATH.read_text(encoding="utf-8"))
            if data.get("prompt_hash") == prompt_hash:
                existing = data.get("entries", {})
        except (json.JSONDecodeError, OSError):
            existing = {}
    scored_at = datetime.now(TZ_CN).isoformat()
    for arxiv_id, score in scores.items():
        existing[arxiv_id] = {"score": score, "scored_at": scored_at}
    payload = {"prompt_hash": prompt_hash, "entries": existing}
    PAPER_SCORE_CACHE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_llm_paper_score(response_text: str) -> int:
    match = re.search(r"[1-5]", (response_text or "").strip())
    if not match:
        return FAILED_PAPER_LLM_SCORE
    return int(match.group())


def _mistral_score_paper(title: str, summary: str, api_key: str) -> int:
    user_prompt = _PAPER_LLM_USER_TEMPLATE.format(
        title=title,
        summary=(summary or "")[:300],
    )
    response = _mistral_client(api_key).chat.complete(
        model=PAPER_LLM_MODEL,
        messages=[
            {"role": "system", "content": _PAPER_LLM_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=8,
    )
    content = response.choices[0].message.content
    return _parse_llm_paper_score(content)


def _score_single_paper(item: Dict, api_key: str) -> int:
    try:
        return _mistral_score_paper(
            item.get("title", "") or "",
            item.get("summary", "") or "",
            api_key,
        )
    except Exception:
        return FAILED_PAPER_LLM_SCORE


def score_papers_with_llm(items: List[Dict]) -> None:
    """Score arXiv papers via Mistral; reuse disk cache keyed by arXiv ID + prompt hash."""
    global _paper_llm_scores

    api_key = _load_mistral_api_key()
    papers = [item for item in items if item.get("source") == "arxiv"]
    prompt_hash = _paper_prompt_version_hash()
    disk_cache = _load_paper_score_cache(prompt_hash)
    cache_hits = 0

    unique_papers: Dict[str, Dict] = {}
    for paper in papers:
        cache_key = _paper_cache_key(paper)
        if cache_key in _paper_llm_scores:
            continue
        arxiv_id = _arxiv_id(paper)
        if arxiv_id and arxiv_id in disk_cache:
            _paper_llm_scores[cache_key] = disk_cache[arxiv_id]
            cache_hits += 1
            continue
        unique_papers.setdefault(cache_key, paper)

    if cache_hits:
        print(f"Paper score cache hits: {cache_hits}/{len(papers)} (prompt {prompt_hash})")

    if not unique_papers:
        if papers:
            qualifying = sum(1 for item in papers if _is_qualifying_paper(item))
            print(
                f"Paper LLM scoring complete: {qualifying}/{len(papers)} qualifying "
                f"(hybrid threshold {MIN_PAPER_LLM_SCORE}+ or score 3 w/ "
                f"importance>={MIN_PAPER_HYBRID_IMPORTANCE})"
            )
            _save_paper_scores_debug(papers)
        return

    if not _HAS_MISTRAL:
        print("Warning: mistralai SDK not installed; defaulting all paper scores to 3")
        for key in unique_papers:
            _paper_llm_scores[key] = FAILED_PAPER_LLM_SCORE
        return

    if not api_key:
        print("Warning: MISTRAL_API_KEY not set; defaulting all paper scores to 3")
        for key in unique_papers:
            _paper_llm_scores[key] = FAILED_PAPER_LLM_SCORE
        return

    print(f"Scoring {len(unique_papers)} uncached papers with {PAPER_LLM_MODEL}...")
    new_scores: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=PAPER_LLM_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_score_single_paper, paper, api_key): (cache_key, paper)
            for cache_key, paper in unique_papers.items()
        }
        for future in as_completed(futures):
            cache_key, paper = futures[future]
            try:
                score = future.result()
            except Exception:
                score = FAILED_PAPER_LLM_SCORE
            _paper_llm_scores[cache_key] = score
            arxiv_id = _arxiv_id(paper)
            if arxiv_id:
                new_scores[arxiv_id] = score

    if new_scores:
        merged = dict(disk_cache)
        merged.update(new_scores)
        _save_paper_score_cache(prompt_hash, merged)

    qualifying = sum(1 for item in papers if _is_qualifying_paper(item))
    print(
        f"Paper LLM scoring complete: {qualifying}/{len(papers)} qualifying "
        f"(hybrid threshold {MIN_PAPER_LLM_SCORE}+ or score 3 w/ "
        f"importance>={MIN_PAPER_HYBRID_IMPORTANCE})"
    )
    _save_paper_scores_debug(papers)


def _save_paper_scores_debug(papers: List[Dict]) -> None:
    """Write per-paper Mistral scores to logs/paper_scores_today.json."""
    records = []
    for paper in papers:
        score = paper_llm_score(paper)
        records.append({
            "title": paper.get("title", ""),
            "score": score,
            "qualified": _is_qualifying_paper(paper),
            "dropped": not _is_qualifying_paper(paper),
            "source_url": paper.get("source_url", ""),
            "arxiv_id": _arxiv_id(paper),
        })
    PAPER_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAPER_SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now(TZ_CN).isoformat(),
                "papers": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Paper score debug log saved to: {PAPER_SCORES_PATH}")


def _parse_json_array_from_llm(text: str) -> List[Dict]:
    """Extract a JSON array from LLM output."""
    if not text:
        raise ValueError("empty LLM response")
    m = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    if not m:
        if re.search(r"\[\s*\]", text):
            return []
        raise ValueError("no JSON array in LLM response")
    return json.loads(m.group(1))


def _verify_digest_with_mistral(
    candidates: List[Dict],
    memory_spine: List[Dict],
    api_key: str,
) -> List[Dict]:
    """Plan-Execute-Verify: one Mistral pass to dedupe against memory spine."""
    slim_candidates = [
        {
            "title": item.get("title", ""),
            "summary": (item.get("summary") or "")[:200],
            "entity": item.get("entity", ""),
            "source": item.get("source", ""),
            "source_url": item.get("source_url", ""),
            "importance": item.get("importance", 0),
        }
        for item in candidates
    ]
    user_prompt = _VERIFY_LLM_USER_TEMPLATE.format(
        top_items_json=json.dumps(slim_candidates, ensure_ascii=False, indent=2),
        memory_spine_json=json.dumps(memory_spine, ensure_ascii=False, indent=2),
    )
    response = _mistral_client(api_key).chat.complete(
        model=VERIFY_LLM_MODEL,
        messages=[
            {"role": "system", "content": _VERIFY_LLM_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=8192,
    )
    content = response.choices[0].message.content
    filtered_slim = _parse_json_array_from_llm(content)

    by_url = {_normalize_source_url(item.get("source_url", "")): item for item in candidates}
    by_title = {(item.get("title") or "").strip().lower(): item for item in candidates}
    verified: List[Dict] = []
    seen: set = set()
    for raw in filtered_slim:
        url_key = _normalize_source_url(raw.get("source_url", ""))
        title_key = (raw.get("title") or "").strip().lower()
        original = by_url.get(url_key) or by_title.get(title_key)
        if not original:
            continue
        key = _item_key(original)
        if key in seen:
            continue
        seen.add(key)
        verified.append(original)
    return verified


def _memory_spine_match_keys(
    memory_spine: List[Dict],
) -> Tuple[Set[str], Set[Tuple[str, str]], Set[str]]:
    """Build URL, (entity, content_type), and repo-key sets from the memory spine."""
    urls: Set[str] = set()
    entity_types: Set[Tuple[str, str]] = set()
    repo_keys: Set[str] = set()
    for entry in memory_spine:
        url = _normalize_source_url(entry.get("source_url", ""))
        if url:
            urls.add(url)
        entity = (entry.get("entity") or "").strip()
        content_type = (entry.get("content_type") or "").strip()
        if entity and content_type:
            entity_types.add((entity, content_type))
        repo_key = _github_repo_key(entry)
        if repo_key:
            repo_keys.add(repo_key)
    return urls, entity_types, repo_keys


def _editorial_verify_items(
    candidates: List[Dict],
    memory_spine: List[Dict],
    exempt_keys: Optional[Set[str]] = None,
) -> List[Dict]:
    """Rule-based editorial verify with source-specific dedup rules."""
    if not memory_spine:
        return candidates
    exempt = exempt_keys or set()
    spine_urls, spine_entity_types, spine_repo_keys = _memory_spine_match_keys(memory_spine)
    verified: List[Dict] = []
    for item in candidates:
        if _item_key(item) in exempt:
            verified.append(item)
            continue

        source = item.get("source", "")
        url = _normalize_source_url(item.get("source_url", ""))

        if _is_ainews(item):
            if url and url in spine_urls:
                continue
            verified.append(item)
            continue

        if source in ("github", "repo"):
            repo_key = _github_repo_key(item)
            if url and url in spine_urls:
                continue
            if repo_key and repo_key in spine_repo_keys:
                continue
            verified.append(item)
            continue

        if source in ("arxiv", "web"):
            if url and url in spine_urls:
                continue
            entity = (item.get("entity") or "").strip()
            content_type = (item.get("content_type") or "").strip()
            if entity and content_type and (entity, content_type) in spine_entity_types:
                continue
            verified.append(item)
            continue

        if url and url in spine_urls:
            continue
        verified.append(item)
    return verified


def _generate_single_description(item: Dict, api_key: str) -> str:
    try:
        response = _mistral_client(api_key).chat.complete(
            model=DESCRIPTION_LLM_MODEL,
            messages=[
                {"role": "system", "content": _DESCRIPTION_LLM_SYSTEM},
                {
                    "role": "user",
                    "content": _DESCRIPTION_LLM_USER_TEMPLATE.format(
                        title=item.get("title", "") or "",
                        summary=(item.get("summary") or "")[:300],
                        source=item.get("source", "") or "",
                    ),
                },
            ],
            temperature=0,
            max_tokens=40,
        )
        text = (response.choices[0].message.content or "").strip()
        text = text.strip('"').strip("'")
        words = text.split()
        if len(words) > 12:
            text = " ".join(words[:12])
        return text
    except Exception:
        summary = (item.get("summary") or "").strip()
        words = summary.split()
        return " ".join(words[:12]) if words else item.get("title", "")


def _generate_digest_descriptions(items: List[Dict], api_key: Optional[str]) -> None:
    """Pre-write one-line digest descriptions on each selected item."""
    if not items:
        return
    if not api_key or not _HAS_MISTRAL:
        for item in items:
            summary = (item.get("summary") or "").strip()
            words = summary.split()
            item["digest_description"] = " ".join(words[:12]) if words else item.get("title", "")
        return

    print(f"Generating digest descriptions for {len(items)} items...")
    with ThreadPoolExecutor(max_workers=PAPER_LLM_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_generate_single_description, item, api_key): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                item["digest_description"] = future.result()
            except Exception:
                summary = (item.get("summary") or "").strip()
                words = summary.split()
                item["digest_description"] = " ".join(words[:12]) if words else item.get("title", "")


def _rejection_reason(
    item: Dict,
    *,
    filtered_items: List[Dict],
    top_items: List[Dict],
    verified_items: List[Dict],
) -> str:
    """Return why an item did not appear in the final digest."""
    key = _item_key(item)
    if key in {_item_key(row) for row in verified_items}:
        return ""

    if key in {_item_key(row) for row in top_items}:
        return "editorial verify dedup"

    _, stale_skipped, _ = filter_stale_items([item])
    if stale_skipped:
        return "stale date filter"

    if _is_skipped_repo_content(item) or _is_skipped_github_trending(item):
        return "repo skip policy"

    if _is_frequency_capped(item):
        count = _repo_delivery_count(item)
        return f"frequency cap ({count}/{REPO_FREQUENCY_CAP} in {REPO_FREQUENCY_LOOKBACK_DAYS}d)"

    if _is_maintenance_repo_release(item):
        return "maintenance/plugin filter"

    if item.get("source") == "arxiv" and not _is_qualifying_paper(item):
        return (
            f"paper LLM score {paper_llm_score(item)} below hybrid threshold "
            f"({MIN_PAPER_LLM_SCORE}+ or 3 w/ >={MIN_PAPER_HYBRID_IMPORTANCE} importance)"
        )

    if not _is_eligible_for_digest(item):
        return "failed eligibility"

    if key not in {_item_key(row) for row in filtered_items}:
        return "delivered-log dedup"

    return "not selected (quota/ranking)"


def _save_rejected_items(
    raw_items: List[Dict],
    *,
    filtered_items: List[Dict],
    top_items: List[Dict],
    verified_items: List[Dict],
) -> None:
    """Save debug log of items not included in the final digest."""
    verified_keys = {_item_key(item) for item in verified_items}
    rejected = []
    for item in raw_items:
        key = _item_key(item)
        if key in verified_keys:
            continue
        reason = _rejection_reason(
            item,
            filtered_items=filtered_items,
            top_items=top_items,
            verified_items=verified_items,
        )
        rejected.append({
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "source_url": item.get("source_url", ""),
            "importance": item.get("importance", 0),
            "rejection_reason": reason or "unknown",
        })
    REJECTED_ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REJECTED_ITEMS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now(TZ_CN).isoformat(),
                "rejected_count": len(rejected),
                "items": rejected,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Rejected items debug log saved to: {REJECTED_ITEMS_PATH} ({len(rejected)} items)")


def _restore_quota_minimums(candidates: List[Dict], verified: List[Dict]) -> List[Dict]:
    """Re-add quota-minimum items removed by LLM verify (e.g. X signals deduped as same story)."""
    verified_keys = {_item_key(item) for item in verified}
    result = list(verified)
    bucket_counts: Dict[str, int] = {}
    for item in result:
        bucket = _quota_bucket(item)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    restored = 0
    for bucket, limits in QUOTA_LIMITS.items():
        minimum = limits["min"]
        if minimum <= 0:
            continue
        shortfall = minimum - bucket_counts.get(bucket, 0)
        if shortfall <= 0:
            continue
        predicate = _bucket_predicate(bucket)
        for item in sorted(candidates, key=lambda x: x.get("importance", 0), reverse=True):
            if shortfall <= 0:
                break
            key = _item_key(item)
            if key in verified_keys or not predicate(item) or not _is_eligible_for_digest(item):
                continue
            result.append(item)
            verified_keys.add(key)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            shortfall -= 1
            restored += 1

    if restored:
        print(f"Restored {restored} quota-minimum item(s) after editorial verify")
    return sorted(result, key=lambda x: x.get("importance", 0), reverse=True)


def _print_top_items_list(label: str, items: List[Dict]) -> None:
    print(label)
    for item in items:
        category = _digest_category(item)
        paper_note = ""
        if item.get("source") == "arxiv":
            paper_note = f" [LLM relevance {paper_llm_score(item)}]"
        print(
            f"  [{item.get('importance', 0):.1f}] {category:22} "
            f"{item.get('title', 'No title')}{paper_note}"
        )


def paper_llm_score(item: Dict) -> int:
    if item.get("source") != "arxiv":
        return 0
    return _paper_llm_scores.get(_paper_cache_key(item), FAILED_PAPER_LLM_SCORE)


def _is_qualifying_paper(item: Dict) -> bool:
    if item.get("source") != "arxiv":
        return False
    score = paper_llm_score(item)
    if score >= MIN_PAPER_LLM_SCORE:
        return True
    return score == FAILED_PAPER_LLM_SCORE and item.get("importance", 0) >= MIN_PAPER_HYBRID_IMPORTANCE


def _is_qualifying_web(item: Dict) -> bool:
    if item.get("source") != "web":
        return False
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    url = (item.get("source_url") or "").lower()
    if any(keyword in text for keyword in _WEB_SKIP_KEYWORDS):
        return False
    if any(domain in url for domain in _WEB_NEWS_DOMAINS):
        return False
    if item.get("entity") == "AI Engineering Web":
        if not any(keyword in text for keyword in _WEB_RELEASE_KEYWORDS):
            return False
    return any(keyword in text for keyword in _WEB_RELEVANCE_KEYWORDS)


def _hf_text(item: Dict) -> str:
    return f"{item.get('title', '')} {item.get('summary', '')} {item.get('raw_text', '')}".lower()


def _is_qualifying_huggingface(item: Dict) -> bool:
    if item.get("source") != "huggingface":
        return False
    text = _hf_text(item)
    if any(keyword in text for keyword in _HF_SKIP_KEYWORDS):
        return False
    return any(keyword in text for keyword in _HF_INCLUDE_KEYWORDS)


def _github_text(item: Dict) -> str:
    return f"{item.get('title', '')} {item.get('summary', '')} {item.get('raw_text', '')}"


def _clean_github_summary(summary: str) -> str:
    text = re.sub(r"##[^\n]*\n", " ", summary)
    text = re.sub(r"\*+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_added_or_new_signal(text: str) -> bool:
    scrubbed = re.sub(r"new contributors?", " ", text, flags=re.I)
    lower = scrubbed.lower()
    return bool(re.search(r"\badded\b", lower) or re.search(r"\bnew\b", lower))


def _first_changelog_entry(summary: str) -> str:
    for line in summary.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line.lstrip("-* ").strip()
    return _clean_github_summary(summary)


def _changelog_bullets(text: str) -> List[str]:
    """Extract markdown changelog bullets, skipping install command lines."""
    bullets: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("-", "*")):
            continue
        bullet = stripped.lstrip("-* ").strip()
        lower = bullet.lower()
        if any(cmd in lower for cmd in ("npm install", "pip install", "yarn add", "pnpm add", "bun add")):
            continue
        bullets.append(bullet)
    return bullets


def _is_claude_agent_sdk_sync_release(item: Dict) -> bool:
    """SDK releases that only sync bundled CLI, internal wiring, or minor telemetry fields."""
    repo_key = _github_repo_key(item)
    if not repo_key or not repo_key.startswith("anthropics/claude-agent-sdk"):
        return False

    text = _github_text(item).lower()
    if any(keyword in text for keyword in ("updated bundled", "bundled claude cli", "updated to parity")):
        return True

    combined = item.get("raw_text") or item.get("summary") or ""
    substantive = _changelog_bullets(combined)
    if not substantive or len(substantive) > 3:
        return False

    for bullet in substantive:
        lower = bullet.lower()
        if any(keyword in lower for keyword in _MAJOR_RELEASE_CAPABILITY_KEYWORDS):
            return False

    if all(bullet.lower().startswith("added") for bullet in substantive):
        return True

    return all(
        bullet.lower().startswith("added")
        and any(keyword in bullet.lower() for keyword in _SDK_SYNC_KEYWORDS)
        for bullet in substantive
    )


def _is_fix_heavy_claude_code_patch(item: Dict) -> bool:
    """Patch releases dominated by bugfixes with only minor CLI UX tweaks."""
    repo_key = _github_repo_key(item)
    if repo_key != "anthropics/claude-code":
        return False
    if not re.search(r"v\d+\.\d+\.\d+", item.get("title", ""), re.I):
        return False

    combined = item.get("raw_text") or item.get("summary") or ""
    bullets = _changelog_bullets(combined)
    if len(bullets) < 8:
        return False

    fixed = [bullet for bullet in bullets if bullet.lower().startswith("fixed")]
    other = [
        bullet for bullet in bullets
        if bullet.lower().startswith(("added", "improved", "changed"))
    ]
    if len(fixed) < max(8, len(other) * 2):
        return False

    for bullet in other:
        lower = bullet.lower()
        if any(keyword in lower for keyword in _MAJOR_RELEASE_CAPABILITY_KEYWORDS):
            return False
    return True


def _is_fixed_only_maintenance(summary: str) -> bool:
    if not summary.strip():
        return False

    bullets = [
        line.strip().lstrip("-* ").strip()
        for line in summary.splitlines()
        if line.strip().startswith(("-", "*"))
    ]
    if len(bullets) == 1 and bullets[0].lower().startswith("fixed"):
        return True

    first = _first_changelog_entry(summary)
    if first.lower().startswith("fixed") and not _has_added_or_new_signal(summary):
        return True

    return False


def _is_short_low_signal_summary(summary: str) -> bool:
    """Skip terse release notes with no feature, capability, or improvement signal."""
    clean = _clean_github_summary(summary).lower()
    if not clean:
        return True
    if len(clean.split()) >= 15:
        return False
    return not any(keyword in clean for keyword in _GITHUB_FEATURE_KEYWORDS)


def _is_github_trending(item: Dict) -> bool:
    return "trending" in (item.get("source_label") or "").lower()


def _github_trending_has_empty_description(item: Dict) -> bool:
    """OSS Insight trending rows often ship without summary/raw_text."""
    return not (item.get("summary") or "").strip() and not (item.get("raw_text") or "").strip()


def _github_trending_slug_text(item: Dict) -> str:
    repo_key = _github_repo_key(item)
    title = item.get("title", "") or ""
    return f"{title} {repo_key}".lower()


def _is_trending_ai_from_slug(item: Dict) -> bool:
    """AI signal from repo slug + title when description is empty."""
    text = _github_trending_slug_text(item)
    return any(keyword in text for keyword in _GITHUB_TRENDING_AI_KEYWORDS)


def _is_curated_repo_source(item: Dict) -> bool:
    """True for GitHub Trending and Good AI List repo entries."""
    if item.get("source") == "repo" and _is_good_ai_list(item):
        return True
    return item.get("source") == "github" and _is_github_trending(item)


def _is_skipped_repo_content(item: Dict) -> bool:
    """Skip blocklisted or low-signal repos (GitHub Trending + Good AI List)."""
    if not _is_curated_repo_source(item):
        return False

    repo_key = _github_repo_key(item)
    if repo_key in _REPO_BLOCKLIST:
        return True

    text = _github_text(item).lower()
    if any(keyword in text for keyword in _REPO_NON_ENGINEERING_KEYWORDS):
        return True
    if any(keyword in text for keyword in _REPO_WEBSITE_CLONER_KEYWORDS):
        return True

    has_ai_signal = any(signal in text for signal in _REPO_AI_AGENT_FRAMEWORK_SIGNALS)
    if any(keyword in text for keyword in _REPO_GENERIC_SCRAPING_KEYWORDS) and not has_ai_signal:
        return True
    if any(keyword in text for keyword in _REPO_BROWSER_AUTOMATION_KEYWORDS) and not has_ai_signal:
        return True
    return False


def _is_skipped_github_cybersecurity(text: str) -> bool:
    """Skip offensive-security cheat-sheet repos; plain cheat-sheets stay."""
    if any(phrase in text for phrase in _GITHUB_TRENDING_CYBERSECURITY_PHRASES):
        return True
    return (
        "command library equipped with" in text
        and "cheat-sheet" in text
    )


def _is_skipped_github_trending(item: Dict) -> bool:
    """Skip finance, plugin, cybersecurity, and low-signal repos from GitHub Trending."""
    if not _is_github_trending(item):
        return False
    if _is_skipped_repo_content(item):
        return True
    text = _github_text(item).lower()
    if _is_skipped_github_cybersecurity(text):
        return True
    if any(keyword in text for keyword in _GITHUB_TRENDING_FINANCE_KEYWORDS):
        return True
    return any(keyword in text for keyword in _GITHUB_TRENDING_PLUGIN_KEYWORDS)


def _is_ai_engineering_repo_text(text: str) -> bool:
    """Return True when repo name/description signals AI engineering."""
    lower = text.lower()
    return any(keyword in lower for keyword in _GITHUB_TRENDING_AI_KEYWORDS)


def _is_low_signal_plugin_repo(text: str) -> bool:
    lower = text.lower()
    if any(keyword in lower for keyword in _GITHUB_LOW_SIGNAL_PLUGIN_KEYWORDS):
        return True
    return any(keyword in lower for keyword in _GITHUB_TRENDING_PLUGIN_KEYWORDS)


def _is_maintenance_repo_release(item: Dict) -> bool:
    """Skip maintenance, plugin-wrapper, and low-signal repo blurbs (GitHub + Good AI List)."""
    source = item.get("source", "")
    is_github = source == "github"
    is_gal = source == "repo" and _is_good_ai_list(item)
    if not is_github and not is_gal:
        return False

    if _is_skipped_repo_content(item):
        return True

    if is_github and _is_skipped_github_trending(item):
        return True

    text = _github_text(item)
    lower = text.lower()
    if _is_low_signal_plugin_repo(lower):
        return True

    summary = item.get("summary", "") or ""
    if is_gal:
        if any(keyword in lower for keyword in _GITHUB_MAINTENANCE_KEYWORDS):
            return True
        return False

    is_trending_style = is_github and _is_github_trending(item)
    if is_trending_style:
        if _github_trending_has_empty_description(item):
            return not _is_trending_ai_from_slug(item)
        return _is_short_low_signal_summary(summary)

    if is_github and not is_trending_style:
        if _is_claude_agent_sdk_sync_release(item):
            return True
        if _is_fix_heavy_claude_code_patch(item):
            return True

    if any(keyword in lower for keyword in _GITHUB_MAINTENANCE_KEYWORDS):
        return True
    if re.search(r"\bCI\b", text):
        return True
    if _is_fixed_only_maintenance(summary):
        return True
    return _is_short_low_signal_summary(summary)


def _is_qualifying_hackernews(item: Dict) -> bool:
    """HN items must signal open-source AI engineering; skip closed-product hype."""
    if item.get("source") != "hackernews":
        return False
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    title = (item.get("title") or "").lower()
    if not any(keyword in text for keyword in _HN_INCLUDE_KEYWORDS):
        return False
    if any(term in text for term in _HN_CLOSED_COMMERCIAL):
        if any(term in title for term in _HN_ENGINEERING_TITLE_TERMS):
            return True
        return any(signal in text for signal in _HN_OPEN_SIGNAL)
    return True


def _is_qualifying_ainews(item: Dict) -> bool:
    if not _is_ainews(item) or _is_ainews_x(item):
        return False
    return bool((item.get("source_url") or "").strip())


def _x_text(item: Dict) -> str:
    return f"{item.get('title', '')} {item.get('summary', '')} {item.get('raw_text', '')}".lower()


def _x_engagement(item: Dict) -> Tuple[int, int]:
    engagement = item.get("engagement") or {}
    return int(engagement.get("views") or 0), int(engagement.get("likes") or 0)


def _x_quality_rejection_reason(item: Dict) -> Optional[str]:
    """Return rejection reason for X-bucket items, or None if qualifying."""
    if item.get("source") != "x" and not _is_ainews_x(item):
        return None
    text = _x_text(item)
    if not any(keyword in text for keyword in _X_QUALITY_KEYWORDS):
        return "x quality: missing engineering keyword"
    if item.get("source") == "x":
        views, likes = _x_engagement(item)
        if views < _X_MIN_VIEWS and likes < _X_MIN_LIKES:
            return f"x quality: low engagement ({views} views, {likes} likes)"
    return None


def _is_qualifying_x(item: Dict) -> bool:
    if item.get("source") != "x" and not _is_ainews_x(item):
        return False
    return _x_quality_rejection_reason(item) is None


def filter_x_quality(items: List[Dict]) -> Tuple[List[Dict], int, List[str]]:
    """Remove X-bucket items that fail engineering keyword or engagement checks."""
    kept: List[Dict] = []
    skipped = 0
    rejected_titles: List[str] = []
    for item in items:
        reason = _x_quality_rejection_reason(item)
        if reason:
            skipped += 1
            rejected_titles.append(item.get("title") or reason)
            continue
        kept.append(item)
    return kept, skipped, rejected_titles


def _is_qualifying_github(item: Dict) -> bool:
    if item.get("source") != "github":
        return True
    if _is_frequency_capped(item):
        return False
    if _is_maintenance_repo_release(item):
        return False
    if _is_github_trending(item):
        if _github_trending_has_empty_description(item):
            return _is_trending_ai_from_slug(item)
        return _is_ai_engineering_repo_text(_github_text(item))
    return True


def _is_qualifying_good_ai_list(item: Dict) -> bool:
    if _is_frequency_capped(item):
        return False
    return _is_good_ai_list(item) and not _is_maintenance_repo_release(item)


def _is_eligible_for_digest(item: Dict) -> bool:
    source = item.get("source")
    if source == "arxiv":
        return _is_qualifying_paper(item)
    if source == "web":
        if _is_ainews(item):
            return _is_qualifying_ainews(item)
        return _is_qualifying_web(item)
    if source == "huggingface":
        return _is_qualifying_huggingface(item)
    if source == "github":
        return _is_qualifying_github(item)
    if _is_good_ai_list(item):
        return _is_qualifying_good_ai_list(item)
    if source == "hackernews":
        return _is_qualifying_hackernews(item)
    if source == "x" or _is_ainews_x(item):
        return _is_qualifying_x(item)
    return True


def _digest_category(item: Dict) -> str:
    source = item.get("source", "")
    label = item.get("source_label", "")

    if source == "arxiv":
        return "Paper"
    if source == "huggingface":
        return "HuggingFace"
    if _is_good_ai_list(item):
        return "Repo (Good AI List)"
    if source == "repo":
        return "Repo"
    if source == "web":
        if _is_ainews_x(item):
            return "Community (X via AINews)"
        if _is_ainews(item):
            return "AINews"
        return "Web (Tavily)"
    if source == "github":
        if "Trending" in label:
            return "Repo (GitHub Trending)"
        return "Repo (GitHub)"
    if source == "hackernews":
        return "Community (HN)"
    if source == "reddit":
        return "Community (Reddit)"
    if source == "x":
        return "x"
    return source or "Other"


def _quota_bucket(item: Dict) -> str:
    """Map an item to its quota bucket for min/max enforcement."""
    if _is_good_ai_list(item):
        return "good_ai_list"
    if _is_ainews_x(item):
        return "x"
    if _is_ainews(item):
        return "ainews"
    source = item.get("source", "")
    return {
        "arxiv": "paper",
        "github": "github",
        "web": "web",
        "huggingface": "huggingface",
        "x": "x",
        "hackernews": "hackernews",
    }.get(source, "other")


def _bucket_predicate(bucket: str) -> Callable[[Dict], bool]:
    if bucket == "paper":
        return _is_qualifying_paper
    if bucket == "github":
        return lambda item: item.get("source") == "github" and _is_qualifying_github(item)
    if bucket == "web":
        return lambda item: (
            item.get("source") == "web"
            and not _is_ainews(item)
            and _is_qualifying_web(item)
        )
    if bucket == "ainews":
        return _is_qualifying_ainews
    if bucket == "huggingface":
        return _is_qualifying_huggingface
    if bucket == "good_ai_list":
        return _is_qualifying_good_ai_list
    if bucket == "x":
        return lambda item: _is_qualifying_x(item)
    if bucket == "hackernews":
        return _is_qualifying_hackernews
    return lambda item: False


def _can_add_to_bucket(item: Dict, bucket_counts: Dict[str, int]) -> bool:
    bucket = _quota_bucket(item)
    limits = QUOTA_LIMITS.get(bucket)
    if limits is None:
        return True
    return bucket_counts.get(bucket, 0) < limits["max"]


def _paper_sort_key(item: Dict) -> Tuple[int, float]:
    """Sort paper candidates: undelivered first, then highest score."""
    delivered_rank = 1 if _was_recently_delivered(item) else 0
    return (delivered_rank, -item.get("importance", 0))


def _paper_candidates(items: List[Dict]) -> List[Dict]:
    """Qualifying papers sorted by freshness then score."""
    qualifying = [item for item in items if _is_qualifying_paper(item)]
    return sorted(qualifying, key=_paper_sort_key)


def _good_ai_list_sort_key(item: Dict) -> Tuple[int, float]:
    """Sort Good AI List candidates: fresh first, then highest score."""
    delivered_rank = 1 if _was_recently_delivered(item) else 0
    return (delivered_rank, -item.get("importance", 0))


def _good_ai_list_redelivery_allowed(item: Dict, redelivered_count: int) -> bool:
    """Good AI List repos delivered within lookback may not fill quota slots."""
    if not _was_recently_delivered(item):
        return True
    return redelivered_count < GOOD_AI_LIST_MAX_REDELIVERED


def _good_ai_list_candidates(items: List[Dict]) -> List[Dict]:
    """Qualifying Good AI List repos sorted by freshness then score."""
    qualifying = [
        item for item in items
        if _is_qualifying_good_ai_list(item) and not _was_recently_delivered(item)
    ]
    return sorted(qualifying, key=_good_ai_list_sort_key)


def select_diverse_items(items: List[Dict], top_n: int = 10) -> Tuple[List[Dict], Set[str]]:
    """Select top items with per-category min/max quotas.

    Minimums (when qualifying items exist):
    - Papers: 2, Good AI List: 3, GitHub: 2, AINews: 2, Web/Tavily: 1
    - HN: 1, X: 1, HuggingFace: 0

    Maximums enforced throughout selection:
    - Papers: 4, GitHub: 4, AINews: 4, Web/Tavily: 3, HN: 2, X: 2
    - HuggingFace: 2, Good AI List: 6 (fresh repos only; no redelivery within 14d)
    """
    sorted_items = sorted(items, key=lambda x: x.get("importance", 0), reverse=True)
    paper_candidates = _paper_candidates(items)
    gal_candidates = _good_ai_list_candidates(items)
    gal_redelivered_selected = 0
    ainews_candidates = sorted(
        [item for item in items if _is_qualifying_ainews(item)],
        key=lambda x: x.get("importance", 0),
        reverse=True,
    )
    selected: List[Dict] = []
    selected_keys: set = set()
    quota_minimum_keys: Set[str] = set()
    bucket_counts: Dict[str, int] = {}

    def _try_add(item: Dict) -> bool:
        if len(selected) >= top_n:
            return False
        key = _item_key(item)
        if key in selected_keys or not _can_add_to_bucket(item, bucket_counts):
            return False
        selected.append(item)
        selected_keys.add(key)
        bucket = _quota_bucket(item)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        return True

    for bucket in (
        "paper",
        "good_ai_list",
        "github",
        "ainews",
        "web",
        "hackernews",
        "x",
        "huggingface",
    ):
        minimum = QUOTA_LIMITS[bucket]["min"]
        if minimum <= 0:
            continue
        predicate = _bucket_predicate(bucket)
        if bucket == "good_ai_list":
            candidate_list = gal_candidates
        elif bucket == "paper":
            candidate_list = paper_candidates
        else:
            candidate_list = sorted_items
        picked = 0
        for item in candidate_list:
            if len(selected) >= top_n or picked >= minimum:
                break
            if not predicate(item):
                continue
            if bucket == "good_ai_list" and not _good_ai_list_redelivery_allowed(
                item, gal_redelivered_selected
            ):
                continue
            if _try_add(item):
                if bucket == "good_ai_list" and _was_recently_delivered(item):
                    gal_redelivered_selected += 1
                quota_minimum_keys.add(_item_key(item))
                picked += 1

    for item in ainews_candidates:
        if len(selected) >= top_n:
            break
        key = _item_key(item)
        if key in selected_keys:
            continue
        _try_add(item)

    for item in sorted_items:
        if len(selected) >= top_n:
            break
        key = _item_key(item)
        if (
            key in selected_keys
            or _is_good_ai_list(item)
            or _is_ainews(item)
            or item.get("source") == "arxiv"
        ):
            continue
        if not _is_eligible_for_digest(item):
            continue
        _try_add(item)

    for item in paper_candidates:
        if len(selected) >= top_n:
            break
        key = _item_key(item)
        if key in selected_keys:
            continue
        _try_add(item)

    for item in gal_candidates:
        if len(selected) >= top_n:
            break
        key = _item_key(item)
        if key in selected_keys:
            continue
        if not _good_ai_list_redelivery_allowed(item, gal_redelivered_selected):
            continue
        if _try_add(item):
            if _was_recently_delivered(item):
                gal_redelivered_selected += 1

    return selected, quota_minimum_keys


def _boost_good_ai_list_scores(items: List[Dict]) -> None:
    """Prioritize curated Good AI List repos in selection ranking."""
    for item in items:
        if not _is_good_ai_list(item):
            continue
        boosted = item.get("importance", 0) + GOOD_AI_LIST_IMPORTANCE_BOOST
        item["importance"] = round(min(10.0, boosted), 1)


def _good_ai_list_item_status(
    item: Dict,
    *,
    filtered_keys: set,
    selected_keys: set,
) -> Tuple[str, float]:
    """Return exclusion reason and effective score for a Good AI List audit line."""
    key = _item_key(item)
    base_score = item.get("importance", 0)
    boosted_score = round(min(10.0, base_score + GOOD_AI_LIST_IMPORTANCE_BOOST), 1)

    fresh, stale_skipped, _ = filter_stale_items([item])
    if stale_skipped:
        return "excluded: stale date filter", base_score

    recent = _load_recent_delivered_urls()
    kept, dedup_skipped = filter_already_delivered(fresh, recent)
    if dedup_skipped and not _is_good_ai_list(item):
        return "excluded: delivered-log dedup (14d)", base_score

    if _is_skipped_repo_content(item):
        return "excluded: repo skip policy", base_score

    if _is_frequency_capped(item):
        count = _repo_delivery_count(item)
        return f"excluded: frequency cap ({count}/{REPO_FREQUENCY_CAP} in {REPO_FREQUENCY_LOOKBACK_DAYS}d)", base_score

    if _is_maintenance_repo_release(item):
        return "excluded: maintenance/plugin filter", base_score

    if not _is_qualifying_good_ai_list(item):
        return "excluded: failed eligibility", base_score

    if _was_recently_delivered(item):
        return "excluded: delivered-log dedup (14d)", base_score

    if key not in filtered_keys:
        return "excluded: removed before selection", boosted_score

    if key in selected_keys:
        freshness = "previously delivered" if _was_recently_delivered(item) else "fresh"
        return f"SELECTED ({freshness})", boosted_score

    return "eligible: fresh (priority)", boosted_score


def _print_good_ai_list_audit(
    raw_items: List[Dict],
    filtered_items: List[Dict],
    selected_items: List[Dict],
) -> None:
    """Print all Good AI List items with scores and keep/exclude reasons."""
    gal_items = [item for item in raw_items if _is_good_ai_list(item)]
    if not gal_items:
        print("Good AI List audit: no items in collection")
        return

    filtered_keys = {_item_key(item) for item in filtered_items}
    selected_keys = {_item_key(item) for item in selected_items}

    print(f"Good AI List audit ({len(gal_items)} collected):")
    print(f"  {'Score':>6}  {'Boost':>6}  Status")
    for item in sorted(gal_items, key=lambda row: -row.get("importance", 0)):
        base = item.get("importance", 0)
        boosted = round(min(10.0, base + GOOD_AI_LIST_IMPORTANCE_BOOST), 1)
        status, _ = _good_ai_list_item_status(
            item,
            filtered_keys=filtered_keys,
            selected_keys=selected_keys,
        )
        title = (item.get("title") or "")[:48]
        print(f"  {base:6.1f}  {boosted:6.1f}  {status:<40} {title}")

    selected_gal = sum(1 for item in gal_items if _item_key(item) in selected_keys)
    print(f"Good AI List selected: {selected_gal}/{len(gal_items)}")
    _print_good_ai_list_selection_breakdown(selected_items)


def _print_good_ai_list_selection_breakdown(selected_items: List[Dict]) -> None:
    """Show which selected Good AI List repos were fresh vs previously delivered."""
    gal_selected = [item for item in selected_items if _is_good_ai_list(item)]
    if not gal_selected:
        print("Good AI List selection breakdown: none selected")
        return

    fresh: List[Dict] = []
    redelivered: List[Dict] = []
    for item in gal_selected:
        if _was_recently_delivered(item):
            redelivered.append(item)
        else:
            fresh.append(item)

    print(f"Good AI List selection breakdown: {len(fresh)} fresh, {len(redelivered)} previously delivered")
    if fresh:
        print("  Fresh (priority):")
        for item in fresh:
            print(f"    [{item.get('importance', 0):.1f}] {item.get('title', '')}")
    if redelivered:
        print("  Previously delivered (fallback):")
        for item in redelivered:
            entry = _recent_delivery_entry(item)
            delivered_on = entry.get("date", "?") if entry else "?"
            print(f"    [{item.get('importance', 0):.1f}] {item.get('title', '')} (last: {delivered_on})")


def _audit_line(audit: bool, message: str) -> None:
    if audit:
        print(f"AUDIT: {message}")


def _source_counts(items: List[Dict]) -> Counter:
    return Counter(item.get("source") or "unknown" for item in items)


def _stage_removal_report(label: str, before: List[Dict], after: List[Dict]) -> str:
    before_c = _source_counts(before)
    after_c = _source_counts(after)
    lines = [
        f"### {label}: {len(before)} -> {len(after)} (removed {len(before) - len(after)})",
        f"  {'source':<14} {'before':>7} {'after':>7} {'removed':>7}",
    ]
    for src in sorted(set(before_c) | set(after_c)):
        b, a = before_c.get(src, 0), after_c.get(src, 0)
        lines.append(f"  {src:<14} {b:>7} {a:>7} {b - a:>7}")
    return "\n".join(lines)


def _count_bucket_qualifying(items: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for bucket in QUOTA_LIMITS:
        predicate = _bucket_predicate(bucket)
        counts[bucket] = sum(1 for item in items if predicate(item))
    return counts


def _count_ineligible_by_reason(items: List[Dict]) -> Dict[str, Counter]:
    """Per-source ineligibility reasons after frequency cap."""
    by_source: Dict[str, Counter] = {}
    for item in items:
        if _is_eligible_for_digest(item):
            continue
        source = item.get("source") or "unknown"
        if source == "arxiv":
            score = paper_llm_score(item)
            reason = f"paper score {score}"
        elif _is_maintenance_repo_release(item):
            reason = "maintenance/plugin"
        elif _is_skipped_repo_content(item) or _is_skipped_github_trending(item):
            reason = "repo skip policy"
        elif source == "web":
            reason = "web/ainews filter"
        elif source == "huggingface":
            reason = "hf keyword filter"
        elif source == "hackernews":
            reason = "hn keyword filter"
        elif source == "x" or _is_ainews_x(item):
            reason = _x_quality_rejection_reason(item) or "x quality filter"
        else:
            reason = "other eligibility"
        by_source.setdefault(source, Counter())[reason] += 1
    return by_source


def _paper_score_distribution(items: List[Dict]) -> Tuple[Dict[int, int], int]:
    """Score histogram (1-5) and qualifying count for arxiv items."""
    dist: Dict[int, int] = {score: 0 for score in range(1, 6)}
    dist[FAILED_PAPER_LLM_SCORE] = 0
    qualifying = 0
    for item in items:
        if item.get("source") != "arxiv":
            continue
        score = paper_llm_score(item)
        dist[score] = dist.get(score, 0) + 1
        if _is_qualifying_paper(item):
            qualifying += 1
    return dist, qualifying


def _bucket_fill_after_selection(items: List[Dict], top_n: int) -> Tuple[Dict[str, int], int, Set[str]]:
    selected, quota_keys = select_diverse_items(items, top_n)
    filled: Dict[str, int] = {}
    for item in selected:
        bucket = _quota_bucket(item)
        filled[bucket] = filled.get(bucket, 0) + 1
    return filled, len(selected), quota_keys


def run_filter_debug_trace(raw_items: List[Dict], top_n: int = 30) -> str:
    """Run pipeline stages and return a full per-source debug trace."""
    _load_morning_ai_env()
    lines: List[str] = [
        "=" * 72,
        f"FILTER DEBUG TRACE — top_n={top_n}, items_in={len(raw_items)}",
        f"generated {datetime.now(TZ_CN).strftime('%Y-%m-%d %H:%M:%S')} UTC+8",
        "=" * 72,
        "",
        "## 1. RAW INPUT BY SOURCE (before any filtering)",
    ]
    raw_c = _source_counts(raw_items)
    lines.append(f"  total: {len(raw_items)}")
    for src, count in sorted(raw_c.items()):
        lines.append(f"  {src}: {count}")

    items = list(raw_items)
    score_papers_with_llm(items)

    papers = [i for i in items if i.get("source") == "arxiv"]
    dist, paper_qual = _paper_score_distribution(items)
    lines.extend([
        "",
        "## 2. PAPER LLM SCORING (all arxiv in raw input)",
        f"  arxiv total: {len(papers)}",
        f"  qualifying (hybrid): {paper_qual}",
        "  score distribution:",
    ])
    for score in sorted(dist):
        if dist[score]:
            label = "score-3 (hybrid eligible if importance>=4.5)" if score == FAILED_PAPER_LLM_SCORE else f"score {score}"
            lines.append(f"    {label}: {dist[score]}")

    stage_before = items
    items, _, _ = filter_stale_items(items)
    lines.extend(["", "## 3. PER-SOURCE SURVIVAL BY STAGE", "", _stage_removal_report("Date filter", stage_before, items)])

    recent_delivered = _load_recent_delivered_urls()
    stage_before = items
    items, _ = filter_already_delivered(items, recent_delivered)
    lines.extend(["", _stage_removal_report("Delivered-log dedup", stage_before, items)])

    stage_before = items
    items, _, _ = filter_frequency_capped(items)
    lines.extend(["", _stage_removal_report("Frequency cap", stage_before, items)])

    stage_before = items
    items, _, _ = filter_x_quality(items)
    lines.extend(["", _stage_removal_report("X quality filter", stage_before, items)])

    post_freq = list(items)
    eligible = [i for i in post_freq if _is_eligible_for_digest(i)]
    ineligible = len(post_freq) - len(eligible)
    lines.extend([
        "",
        f"### Eligibility (maintenance + source rules): {len(post_freq)} -> {len(eligible)} eligible ({ineligible} ineligible)",
        f"  {'source':<14} {'pool':>7} {'eligible':>9} {'blocked':>8}",
    ])
    pool_c = _source_counts(post_freq)
    elig_c = _source_counts(eligible)
    for src in sorted(pool_c):
        pool_n = pool_c[src]
        elig_n = elig_c.get(src, 0)
        lines.append(f"  {src:<14} {pool_n:>7} {elig_n:>9} {pool_n - elig_n:>8}")

    inelig_by_source = _count_ineligible_by_reason(post_freq)
    if inelig_by_source:
        lines.append("")
        lines.append("  Ineligibility reasons by source:")
        for src in sorted(inelig_by_source):
            for reason, count in inelig_by_source[src].most_common():
                lines.append(f"    {src}: {count} x {reason}")

    hn_pool = [i for i in post_freq if i.get("source") == "hackernews"]
    hn_elig = sum(1 for i in hn_pool if _is_qualifying_hackernews(i))
    lines.extend([
        "",
        f"### HN keyword filter: {len(hn_pool)} in pool -> {hn_elig} qualifying",
    ])

    _boost_good_ai_list_scores(post_freq)
    bucket_available = _count_bucket_qualifying(post_freq)
    filled, selected_count, quota_keys = _bucket_fill_after_selection(post_freq, top_n)

    lines.extend([
        "",
        "## 4. QUOTA BUCKET ANALYSIS",
        f"  target top_n: {top_n}",
        f"  {'bucket':<14} {'min':>4} {'max':>4} {'available':>10} {'filled':>7} {'gap_min':>8}",
    ])
    min_unfilled = 0
    max_exhausted = 0
    for bucket, limits in QUOTA_LIMITS.items():
        avail = bucket_available.get(bucket, 0)
        fill = filled.get(bucket, 0)
        gap_min = max(0, limits["min"] - fill)
        if gap_min and avail < limits["min"]:
            min_unfilled += gap_min
        if fill >= limits["max"] and selected_count < top_n:
            max_exhausted += 1
        lines.append(
            f"  {bucket:<14} {limits['min']:>4} {limits['max']:>4} {avail:>10} {fill:>7} {gap_min:>8}"
        )

    lines.extend([
        "",
        f"  quota selection result: {selected_count} / {top_n} slots filled",
        f"  quota-minimum keys (verify-exempt): {len(quota_keys)}",
    ])

    selected, _ = select_diverse_items(post_freq, top_n)
    memory_spine = _load_memory_spine(VERIFY_MEMORY_LOOKBACK_DAYS)
    verified = _editorial_verify_items(selected, memory_spine, exempt_keys=quota_keys)
    removed = len(selected) - len(verified)
    lines.extend([
        "",
        "## 5. EDITORIAL VERIFY",
        f"  memory spine entries: {len(memory_spine)}",
        f"  before verify: {len(selected)}",
        f"  after verify: {len(verified)} (removed {removed}, {len(quota_keys)} exempt)",
    ])
    if removed:
        removed_keys = {_item_key(i) for i in selected} - {_item_key(i) for i in verified}
        lines.append("  removed items:")
        for item in selected:
            if _item_key(item) in removed_keys:
                lines.append(f"    [{_digest_category(item)}] {item.get('title', '')[:80]}")

    lines.extend([
        "",
        "## 6. ROOT CAUSE SUMMARY",
    ])

    shortfall_quota = top_n - selected_count
    shortfall_final = top_n - len(verified)
    eligible_total = len(eligible)

    causes: List[str] = []
    if shortfall_quota > 0:
        causes.append(
            f"Quota selection filled only {selected_count}/{top_n} "
            f"({shortfall_quota} unfilled) — eligible pool has {eligible_total} items"
        )
    for bucket, limits in QUOTA_LIMITS.items():
        avail = bucket_available.get(bucket, 0)
        if limits["min"] > 0 and avail < limits["min"]:
            causes.append(f"Bucket '{bucket}' needs min {limits['min']} but only {avail} qualify")
    zero_buckets = [b for b, c in bucket_available.items() if c == 0 and QUOTA_LIMITS[b]["min"] > 0]
    if zero_buckets:
        causes.append(f"Buckets with zero qualifying items: {', '.join(zero_buckets)}")
    if removed > 0:
        causes.append(f"Editorial verify removed {removed} items ({len(selected)} -> {len(verified)})")

    ainews_avail = bucket_available.get("ainews", 0)
    ainews_fill = filled.get("ainews", 0)
    ainews_max = QUOTA_LIMITS["ainews"]["max"]
    if ainews_fill < ainews_max and ainews_avail > ainews_fill:
        causes.append(
            f"AINews fill gap: {ainews_avail} qualify but only {ainews_fill}/{ainews_max} selected "
            f"(no overflow pass beyond quota minimum)"
        )

    if not causes:
        causes.append("Pipeline reached top_n; no major bottleneck detected.")

    for cause in causes:
        lines.append(f"  - {cause}")

    lines.extend([
        "",
        f"FINAL: {len(verified)} items in digest (target {top_n}, shortfall {shortfall_final})",
        "=" * 72,
    ])
    return "\n".join(lines)


def preview_digest_assessment(item: Dict) -> str:
    """Predict why a raw collection item would or would not reach the digest."""
    fresh, stale_skipped, _ = filter_stale_items([item])
    if stale_skipped:
        return "would drop: stale date filter"

    if _is_frequency_capped(item):
        count = _repo_delivery_count(item)
        return f"would drop: frequency cap ({count}/{REPO_FREQUENCY_CAP} in {REPO_FREQUENCY_LOOKBACK_DAYS}d)"

    if _is_maintenance_repo_release(item):
        return "would drop: maintenance/plugin filter"

    if _is_skipped_repo_content(item) or _is_skipped_github_trending(item):
        return "would drop: repo skip policy"

    if item.get("source") == "arxiv":
        score = paper_llm_score(item)
        if not _is_qualifying_paper(item):
            if score < MIN_PAPER_LLM_SCORE and score != FAILED_PAPER_LLM_SCORE:
                return f"would drop: paper LLM score {score} < {MIN_PAPER_LLM_SCORE}"
            if score == FAILED_PAPER_LLM_SCORE and item.get("importance", 0) < MIN_PAPER_HYBRID_IMPORTANCE:
                return (
                    f"would drop: paper LLM score 3 with importance "
                    f"{item.get('importance', 0):.1f} < {MIN_PAPER_HYBRID_IMPORTANCE}"
                )
            return f"would drop: paper LLM score {score} below threshold"
        if _was_recently_delivered(item):
            return f"eligible paper (score {score}); may compete if quota needs minimum"
        return f"strong paper candidate (LLM score {score})"

    recent = _load_recent_delivered_urls()
    kept, dedup_skipped = filter_already_delivered([item], recent)
    if dedup_skipped and item.get("source") not in ("arxiv",):
        lookback = _delivered_lookback_days(item)
        return f"would drop: delivered-log dedup ({lookback}d window)"

    if not _is_eligible_for_digest(item):
        return "would drop: failed source eligibility rules"

    if _is_good_ai_list(item):
        return "strong Good AI List candidate (fresh, boosted in ranking)"

    if _was_recently_delivered(item):
        return "eligible but previously delivered; lower selection priority"

    return "eligible candidate; competes on importance + quota limits"


def _audit_eligibility_breakdown(items: List[Dict]) -> Dict[str, int]:
    """Count ineligibility reasons in the post-dedup candidate pool."""
    counts: Dict[str, int] = {}
    for item in items:
        if _is_eligible_for_digest(item):
            continue
        source = item.get("source", "")
        if source == "arxiv":
            reason = (
                f"paper LLM score {paper_llm_score(item)} below hybrid threshold "
                f"({MIN_PAPER_LLM_SCORE}+ or 3 w/ >={MIN_PAPER_HYBRID_IMPORTANCE} importance)"
            )
        elif _is_maintenance_repo_release(item):
            reason = "maintenance/plugin filter"
        elif _is_skipped_repo_content(item) or _is_skipped_github_trending(item):
            reason = "repo skip policy"
        elif source == "web":
            reason = "web relevance filter"
        elif source == "huggingface":
            reason = "huggingface keyword filter"
        elif source == "hackernews":
            reason = "hackernews relevance filter"
        else:
            reason = "failed eligibility"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def analyze_collection(input_path: str, top_per_source: int = 10) -> None:
    """Print per-source collection audit with top-N items and digest predictions."""
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with input_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = data.get("items", [])
    if not items:
        print("No items found in data file")
        sys.exit(1)

    _load_morning_ai_env()
    score_papers_with_llm(items)

    by_source: Dict[str, List[Dict]] = {}
    for item in items:
        source = item.get("source", "unknown")
        by_source.setdefault(source, []).append(item)

    print("=" * 72)
    print(f"COLLECTION AUDIT: {input_file.name} ({len(items)} items, date={data.get('date', '?')})")
    print("=" * 72)

    for source in sorted(by_source):
        source_items = sorted(by_source[source], key=lambda row: -row.get("importance", 0))
        print(f"\n## SOURCE: {source} ({len(source_items)} items)")
        print("-" * 72)
        for rank, item in enumerate(source_items[:top_per_source], start=1):
            label = item.get("source_label", "")
            assessment = preview_digest_assessment(item)
            title = (item.get("title") or "No title")[:90]
            print(
                f"  {rank:2}. [{item.get('importance', 0):.1f}] {title}\n"
                f"      label={label or '-'} | {assessment}"
            )
        if len(source_items) > top_per_source:
            print(f"  ... and {len(source_items) - top_per_source} more items")

    stats = data.get("stats", {}).get("by_source", {})
    if stats:
        print("\n## RAW COLLECTOR COUNTS (from stats.by_source)")
        for source in sorted(stats):
            info = stats[source]
            print(f"  {source}: {info.get('items', 0)} items")


def filter_top_items(
    input_path: str,
    output_path: str,
    top_n: int = 30,
    audit: bool = False,
    debug: bool = False,
):
    """Read the full collection JSON and output diverse top-N scored items."""
    if debug:
        audit = True

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items = data.get("items", [])
    items = list(raw_items)

    if not items:
        print("No items found in data file")
        sys.exit(1)

    if debug:
        trace = run_filter_debug_trace(copy.deepcopy(raw_items), top_n)
        FILTER_DEBUG_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FILTER_DEBUG_TRACE_PATH.write_text(trace + "\n", encoding="utf-8")
        print(f"Debug trace saved to: {FILTER_DEBUG_TRACE_PATH}\n")

    print(f"Total items before filtering: {len(items)}")
    _audit_line(audit, f"stage 0 input items: {len(items)}")
    if audit:
        raw_c = _source_counts(items)
        print(f"AUDIT: raw by source — {dict(sorted(raw_c.items()))}")

    _load_morning_ai_env()
    score_papers_with_llm(items)

    papers = [item for item in items if item.get("source") == "arxiv"]
    paper_below = sum(1 for item in papers if not _is_qualifying_paper(item))
    paper_qualifying = sum(1 for item in papers if _is_qualifying_paper(item))
    if audit:
        print(
            f"AUDIT: paper LLM scoring — {len(papers)} arxiv items, "
            f"{paper_qualifying} qualifying (hybrid), "
            f"{paper_below} below threshold"
        )

    pre_date_count = len(items)
    items, stale_count, source_counts = filter_stale_items(items)
    _audit_line(
        audit,
        f"stage 1 date filter — {pre_date_count} -> {len(items)} "
        f"(removed {stale_count} stale)",
    )
    print(f"Skipped {stale_count} stale items (source-specific date cutoffs)")
    print("Items per source after date filter:")
    for source in sorted(source_counts):
        print(f"  {source}: {source_counts[source]}")
    if audit:
        print(f"AUDIT: after date filter by source — {dict(sorted(source_counts.items()))}")

    recent_delivered = _load_recent_delivered_urls()
    match_total, match_url, match_repo = _count_delivered_candidate_matches(items)
    print(
        f"Delivered-log matches among candidates before selection: "
        f"{match_total} total ({match_url} by URL, {match_repo} by GitHub repo key)"
    )
    pre_dedup_count = len(items)
    items, dedup_count = filter_already_delivered(items, recent_delivered)
    _audit_line(
        audit,
        f"stage 2 delivered-log dedup — {pre_dedup_count} -> {len(items)} "
        f"(removed {dedup_count})",
    )
    if audit:
        print(f"AUDIT: after dedup by source — {dict(sorted(_source_counts(items).items()))}")
    gal_redelivered_blocked = sum(
        1 for item in raw_items
        if _is_good_ai_list(item) and _was_recently_delivered(item)
    )
    paper_redeliverable = sum(
        1 for item in items
        if item.get("source") == "arxiv" and _was_recently_delivered(item)
    )
    if gal_redelivered_blocked:
        print(
            f"Good AI List: blocked {gal_redelivered_blocked} previously-delivered repos "
            f"(within {GITHUB_REPO_DELIVERED_LOOKBACK_DAYS}d lookback)"
        )
    if paper_redeliverable:
        print(
            f"Papers: retained {paper_redeliverable} previously-delivered papers "
            f"for freshness-priority minimum quota"
        )
    print(
        f"Skipped {dedup_count} already-delivered items "
        f"(Good AI List: {GITHUB_REPO_DELIVERED_LOOKBACK_DAYS}d, "
        f"GitHub releases: {GITHUB_REPO_DELIVERED_LOOKBACK_DAYS}d repo+semver, "
        f"GitHub Trending: {GITHUB_TRENDING_DELIVERED_LOOKBACK_DAYS}d, "
        f"AINews: {AINEWS_DELIVERED_LOOKBACK_DAYS}d, "
        f"HN Show/Launch: {HN_SHOW_LAUNCH_DELIVERED_LOOKBACK_DAYS}d, "
        f"others: {DEFAULT_DELIVERED_LOOKBACK_DAYS}d)"
    )

    pre_freq_count = len(items)
    items, freq_skipped, freq_capped = filter_frequency_capped(items)
    _audit_line(
        audit,
        f"stage 3 frequency cap — {pre_freq_count} -> {len(items)} "
        f"(removed {freq_skipped})",
    )
    if audit and freq_capped:
        for title in sorted(set(freq_capped)):
            print(f"AUDIT:   frequency-capped: {title}")
    if audit:
        print(f"AUDIT: after frequency cap by source — {dict(sorted(_source_counts(items).items()))}")
    if freq_skipped:
        print(
            f"Skipped {freq_skipped} repos at frequency cap "
            f"({REPO_FREQUENCY_CAP}+ deliveries in {REPO_FREQUENCY_LOOKBACK_DAYS}d):"
        )
        for title in sorted(set(freq_capped)):
            print(f"  - {title}")

    pre_x_quality_count = len(items)
    items, x_quality_skipped, x_quality_rejected = filter_x_quality(items)
    _audit_line(
        audit,
        f"stage 3b x quality filter — {pre_x_quality_count} -> {len(items)} "
        f"(removed {x_quality_skipped})",
    )
    if x_quality_skipped:
        print(f"Skipped {x_quality_skipped} X items (engineering keyword / engagement filter):")
        for title in sorted(set(x_quality_rejected)):
            print(f"  - {title[:80]}")
    if audit and x_quality_rejected:
        for title in sorted(set(x_quality_rejected)):
            print(f"AUDIT:   x-quality-rejected: {title[:80]}")

    if audit:
        eligibility = _audit_eligibility_breakdown(items)
        eligible_count = sum(1 for item in items if _is_eligible_for_digest(item))
        print(
            f"AUDIT: stage 4 eligibility in remaining pool — "
            f"{eligible_count} eligible, {len(items) - eligible_count} ineligible"
        )
        print(f"AUDIT: eligible by source — {dict(sorted(_source_counts([i for i in items if _is_eligible_for_digest(i)]).items()))}")
        bucket_avail = _count_bucket_qualifying(items)
        print("AUDIT: qualifying per quota bucket — " + ", ".join(
            f"{b}={bucket_avail[b]}" for b in QUOTA_LIMITS
        ))
        for reason, count in sorted(eligibility.items(), key=lambda row: -row[1]):
            print(f"AUDIT:   ineligible: {count} x {reason}")

    _boost_good_ai_list_scores(items)

    top_items, quota_minimum_keys = select_diverse_items(items, top_n)
    if quota_minimum_keys:
        print(f"Quota-minimum items protected from editorial verify: {len(quota_minimum_keys)}")
    _audit_line(
        audit,
        f"stage 5 quota selection — selected {len(top_items)} of top {top_n} requested",
    )
    if audit:
        filled: Dict[str, int] = {}
        for item in top_items:
            b = _quota_bucket(item)
            filled[b] = filled.get(b, 0) + 1
        print("AUDIT: quota bucket fill — " + ", ".join(
            f"{b}={filled.get(b, 0)}/{QUOTA_LIMITS[b]['max']}" for b in QUOTA_LIMITS
        ))

    _print_good_ai_list_audit(data.get("items", []), items, top_items)

    _print_top_items_list(f"Top {len(top_items)} items BEFORE editorial verify:", top_items)

    memory_spine = _load_memory_spine(VERIFY_MEMORY_LOOKBACK_DAYS)
    verified_items = top_items
    verify_api_key = _load_mistral_api_key()
    pre_verify_count = len(top_items)
    if memory_spine:
        print(f"Running editorial verify against {len(memory_spine)} memory-spine entries...")
        verified_items = _editorial_verify_items(
            top_items,
            memory_spine,
            exempt_keys=quota_minimum_keys,
        )
        removed = pre_verify_count - len(verified_items)
        _audit_line(
            audit,
            f"stage 6 editorial verify — {pre_verify_count} -> {len(verified_items)} "
            f"(removed {removed}, {len(quota_minimum_keys)} exempt)",
        )
        print(f"Editorial verify removed {removed} item(s)")
    else:
        print("No memory spine entries; skipping editorial verify step")
        _audit_line(audit, "stage 6 editorial verify — skipped (empty memory spine)")

    _audit_line(audit, f"stage 7 final digest count: {len(verified_items)}")

    _print_top_items_list(f"Top {len(verified_items)} items AFTER editorial verify:", verified_items)

    _generate_digest_descriptions(verified_items, verify_api_key)

    print("Category counts:")
    for category, count in sorted(_category_counts(verified_items).items()):
        print(f"  {category}: {count}")

    generated_at = datetime.now(timezone.utc).isoformat()
    output_data = {
        "date": data.get("date", datetime.now().strftime("%Y-%m-%d")),
        "generated_at": generated_at,
        "items": verified_items,
        "stats": {
            "total_collected": len(raw_items),
            "total_in_digest": len(verified_items),
            "before_verify": len(top_items),
            "removed_by_verify": len(top_items) - len(verified_items),
            "by_category": _category_counts(verified_items),
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    delivery_date = output_data["date"]
    _append_delivered_items(verified_items, delivery_date)
    _save_rejected_items(
        raw_items,
        filtered_items=items,
        top_items=top_items,
        verified_items=verified_items,
    )

    print(f"Filtered digest input saved to: {output_path}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f}KB")

    if debug and FILTER_DEBUG_TRACE_PATH.exists():
        print("\n" + FILTER_DEBUG_TRACE_PATH.read_text(encoding="utf-8"))


def _category_counts(items: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        cat = _digest_category(item)
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Filter top scored items for digest generation")
    parser.add_argument("input", nargs="?", help="Path to the full collection JSON file")
    parser.add_argument("output", nargs="?", help="Path to save the filtered output JSON")
    parser.add_argument("--top", type=int, default=30, help="Number of top items to keep (default: 30)")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print stage-by-stage removal counts during filtering",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Full per-source stage trace + quota analysis (implies --audit); saves to logs/filter_debug_trace.txt",
    )
    parser.add_argument(
        "--analyze-input",
        action="store_true",
        help="Audit raw collection JSON per source (no filter output written)",
    )
    args = parser.parse_args()

    if args.analyze_input:
        if not args.input:
            print("Error: --analyze-input requires an input JSON path", file=sys.stderr)
            sys.exit(1)
        analyze_collection(args.input)
        return

    if not args.input or not args.output:
        parser.error("input and output paths are required unless --analyze-input is set")

    filter_top_items(args.input, args.output, args.top, audit=args.audit, debug=args.debug)


if __name__ == "__main__":
    main()
