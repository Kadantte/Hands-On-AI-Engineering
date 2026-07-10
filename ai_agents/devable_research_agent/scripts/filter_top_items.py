#!/usr/bin/env python3
"""
Filter top scored items from the daily collection JSON.
Outputs a smaller JSON file with only the top N items for digest generation.

Applies category diversity quotas so the digest is not dominated by a single source.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from mistralai.client import Mistral

    _HAS_MISTRAL = True
except ImportError:
    _HAS_MISTRAL = False

MORNING_AI_ENV_FILE = Path.home() / ".config" / "morning-ai" / ".env"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELIVERED_ITEMS_PATH = PROJECT_ROOT / "logs" / "delivered_items.json"
DEFAULT_DELIVERED_LOOKBACK_DAYS = 7
GITHUB_REPO_DELIVERED_LOOKBACK_DAYS = 14
REPO_FREQUENCY_CAP = 2
REPO_FREQUENCY_LOOKBACK_DAYS = 30
VERIFY_MEMORY_LOOKBACK_DAYS = 14
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
MIN_PAPER_LLM_SCORE = 5
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

Your job: remove any item from today's list that is:
1. The same repo, tool, or paper that already appeared in the last 14 days (even if the URL is slightly different)
2. The same story covered by a different source
3. A minor update to something already delivered recently

Return only the filtered list as a JSON array with the same structure. Be strict. Remove duplicates ruthlessly."""

# Per-category min/max quotas for digest diversity
QUOTA_LIMITS = {
    "paper": {"min": 1, "max": 4},
    "github": {"min": 2, "max": 5},
    "web": {"min": 1, "max": 2},
    "x": {"min": 1, "max": 2},
    "hackernews": {"min": 0, "max": 2},
    "huggingface": {"min": 0, "max": 2},
    "good_ai_list": {"min": 8, "max": 12},
}

GOOD_AI_LIST_IMPORTANCE_BOOST = 2.0

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
}

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


def _github_repo_key(item_or_url: Any) -> str:
    """Return canonical owner/repo slug for GitHub dedup, or empty string."""
    if isinstance(item_or_url, dict):
        source = item_or_url.get("source", "")
        url = item_or_url.get("source_url", "")
        title = item_or_url.get("title", "")
        if source not in ("github", "repo") and "github.com/" not in url.lower():
            return ""
        for text in (url, title):
            match = _GITHUB_REPO_RE.search(text or "")
            if match:
                return f"{match.group(1).lower()}/{match.group(2).lower()}"
        return ""
    match = _GITHUB_REPO_RE.search(str(item_or_url or ""))
    if match:
        return f"{match.group(1).lower()}/{match.group(2).lower()}"
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
    """URL dedup window: 14 days for GitHub and Good AI List, 7 for others."""
    source = item.get("source", "")
    if source == "github" or source == "repo":
        return GITHUB_REPO_DELIVERED_LOOKBACK_DAYS
    return DEFAULT_DELIVERED_LOOKBACK_DAYS


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
    entry_by_url: Dict[str, Dict] = {}
    entry_by_repo_key: Dict[str, Dict] = {}
    for entry in entries:
        normalized = _normalize_source_url(entry.get("source_url", ""))
        if normalized:
            entry_by_url[normalized] = entry
        repo_key = _github_repo_key(entry)
        if repo_key:
            entry_by_repo_key[repo_key] = entry

    normalized = _normalize_source_url(item.get("source_url", ""))
    repo_key = _github_repo_key(item)
    entry = entry_by_url.get(normalized) if normalized else None
    if entry is None and repo_key:
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
    """Skip items whose source_url or GitHub repo was already posted within the lookback window.

    Good AI List repos are kept in the candidate pool; freshness is applied at selection time.
    arXiv papers are kept for the same reason so the paper minimum quota can be met.
    """
    entries = _load_delivered_entries()
    if not entries and not delivered_urls:
        return items, 0

    now = datetime.now(TZ_CN)
    entry_by_url: Dict[str, Dict] = {}
    entry_by_repo_key: Dict[str, Dict] = {}
    for entry in entries:
        normalized = _normalize_source_url(entry.get("source_url", ""))
        if normalized:
            entry_by_url[normalized] = entry
        repo_key = _github_repo_key(entry)
        if repo_key:
            entry_by_repo_key[repo_key] = entry

    kept: List[Dict] = []
    skipped = 0
    for item in items:
        if _is_good_ai_list(item):
            kept.append(item)
            continue
        if item.get("source") == "arxiv":
            kept.append(item)
            continue

        normalized = _normalize_source_url(item.get("source_url", ""))
        repo_key = _github_repo_key(item)
        entry = None
        if normalized:
            entry = entry_by_url.get(normalized)
        if entry is None and repo_key:
            entry = entry_by_repo_key.get(repo_key)

        if entry:
            lookback = _delivered_lookback_days(item)
            cutoff = (now - timedelta(days=lookback)).strftime("%Y-%m-%d")
            if entry.get("date", "") >= cutoff:
                skipped += 1
                continue
        elif normalized and normalized in delivered_urls:
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
        if repo_key and repo_key in frequency:
            prev_count = int(frequency[repo_key].get("delivery_count", 0))
        new_count = prev_count + 1 if repo_key else 1

        entries.append({
            "source_url": normalized_url,
            "date": delivery_date,
            "title": title,
            "entity": (item.get("entity") or "").strip(),
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
    """Score arXiv papers concurrently via Mistral and cache results."""
    global _paper_llm_scores

    api_key = _load_mistral_api_key()
    papers = [item for item in items if item.get("source") == "arxiv"]
    unique_papers: Dict[str, Dict] = {}
    for paper in papers:
        key = _paper_cache_key(paper)
        if key not in _paper_llm_scores:
            unique_papers.setdefault(key, paper)

    if not unique_papers:
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

    print(f"Scoring {len(unique_papers)} papers with {PAPER_LLM_MODEL}...")
    with ThreadPoolExecutor(max_workers=PAPER_LLM_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_score_single_paper, paper, api_key): cache_key
            for cache_key, paper in unique_papers.items()
        }
        for future in as_completed(futures):
            cache_key = futures[future]
            try:
                _paper_llm_scores[cache_key] = future.result()
            except Exception:
                _paper_llm_scores[cache_key] = FAILED_PAPER_LLM_SCORE

    qualifying = sum(1 for score in _paper_llm_scores.values() if score >= MIN_PAPER_LLM_SCORE)
    print(f"Paper LLM scoring complete: {qualifying}/{len(_paper_llm_scores)} scored >= {MIN_PAPER_LLM_SCORE}")


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
        print(f"Restored {restored} quota-minimum item(s) after LLM verify")
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
    return paper_llm_score(item) >= MIN_PAPER_LLM_SCORE


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
    """SDK releases that only sync bundled CLI or add internal wiring frames."""
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
    if not any(keyword in text for keyword in _HN_INCLUDE_KEYWORDS):
        return False
    if any(term in text for term in _HN_CLOSED_COMMERCIAL):
        return any(signal in text for signal in _HN_OPEN_SIGNAL)
    return True


def _is_qualifying_github(item: Dict) -> bool:
    if item.get("source") != "github":
        return True
    if _is_frequency_capped(item):
        return False
    if _is_maintenance_repo_release(item):
        return False
    if _is_github_trending(item):
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
        return _is_qualifying_web(item)
    if source == "huggingface":
        return _is_qualifying_huggingface(item)
    if source == "github":
        return _is_qualifying_github(item)
    if _is_good_ai_list(item):
        return _is_qualifying_good_ai_list(item)
    if source == "hackernews":
        return _is_qualifying_hackernews(item)
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
            return "Web (AINews)"
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
        return _is_qualifying_web
    if bucket == "huggingface":
        return _is_qualifying_huggingface
    if bucket == "good_ai_list":
        return _is_qualifying_good_ai_list
    if bucket == "x":
        return lambda item: item.get("source") == "x" or _is_ainews_x(item)
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


def _good_ai_list_candidates(items: List[Dict]) -> List[Dict]:
    """Qualifying Good AI List repos sorted by freshness then score."""
    qualifying = [item for item in items if _is_qualifying_good_ai_list(item)]
    return sorted(qualifying, key=_good_ai_list_sort_key)


def select_diverse_items(items: List[Dict], top_n: int = 10) -> List[Dict]:
    """Select top items with per-category min/max quotas.

    Minimums (when qualifying items exist) — filled before any overflow:
    - Papers: 1, Good AI List: 8, GitHub repos: 2, Web/Tavily: 1, X: 1
    - HuggingFace: 0

    Maximums enforced throughout selection:
    - Papers: 4, GitHub repos: 5, Web/Tavily: 2, X: 2, HN: 2
    - HuggingFace: 2, Good AI List: 12

    Minimum quota phase runs paper first, then other buckets; overflow only after.
    """
    sorted_items = sorted(items, key=lambda x: x.get("importance", 0), reverse=True)
    paper_candidates = _paper_candidates(items)
    gal_candidates = _good_ai_list_candidates(items)
    selected: List[Dict] = []
    selected_keys: set = set()
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

    for bucket in ("paper", "good_ai_list", "github", "web", "x", "huggingface"):
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
            if _try_add(item):
                picked += 1

    for item in sorted_items:
        if len(selected) >= top_n:
            break
        key = _item_key(item)
        if key in selected_keys or _is_good_ai_list(item) or item.get("source") == "arxiv":
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
        _try_add(item)

    return selected


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

    if key not in filtered_keys:
        return "excluded: removed before selection", boosted_score

    if key in selected_keys:
        freshness = "previously delivered" if _was_recently_delivered(item) else "fresh"
        return f"SELECTED ({freshness})", boosted_score

    if _was_recently_delivered(item):
        return "eligible: previously delivered (lower priority)", boosted_score
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


def filter_top_items(input_path: str, output_path: str, top_n: int = 20):
    """Read the full collection JSON and output diverse top-N scored items."""

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])

    if not items:
        print("No items found in data file")
        sys.exit(1)

    print(f"Total items before filtering: {len(items)}")

    _load_morning_ai_env()
    score_papers_with_llm(items)

    items, stale_count, source_counts = filter_stale_items(items)
    print(f"Skipped {stale_count} stale items (source-specific date cutoffs)")
    print("Items per source after date filter:")
    for source in sorted(source_counts):
        print(f"  {source}: {source_counts[source]}")

    recent_delivered = _load_recent_delivered_urls()
    match_total, match_url, match_repo = _count_delivered_candidate_matches(items)
    print(
        f"Delivered-log matches among candidates before selection: "
        f"{match_total} total ({match_url} by URL, {match_repo} by GitHub repo key)"
    )
    items, dedup_count = filter_already_delivered(items, recent_delivered)
    gal_redeliverable = sum(
        1 for item in items if _is_good_ai_list(item) and _was_recently_delivered(item)
    )
    paper_redeliverable = sum(
        1 for item in items
        if item.get("source") == "arxiv" and _was_recently_delivered(item)
    )
    if gal_redeliverable:
        print(
            f"Good AI List: retained {gal_redeliverable} previously-delivered repos "
            f"for freshness-priority selection"
        )
    if paper_redeliverable:
        print(
            f"Papers: retained {paper_redeliverable} previously-delivered papers "
            f"for freshness-priority minimum quota"
        )
    print(
        f"Skipped {dedup_count} already-delivered items "
        f"(GitHub/Good AI List: {GITHUB_REPO_DELIVERED_LOOKBACK_DAYS}d, others: {DEFAULT_DELIVERED_LOOKBACK_DAYS}d)"
    )

    items, freq_skipped, freq_capped = filter_frequency_capped(items)
    if freq_skipped:
        print(
            f"Skipped {freq_skipped} repos at frequency cap "
            f"({REPO_FREQUENCY_CAP}+ deliveries in {REPO_FREQUENCY_LOOKBACK_DAYS}d):"
        )
        for title in sorted(set(freq_capped)):
            print(f"  - {title}")

    _boost_good_ai_list_scores(items)

    top_items = select_diverse_items(items, top_n)

    _print_good_ai_list_audit(data.get("items", []), items, top_items)

    _print_top_items_list(f"Top {len(top_items)} items BEFORE LLM verify:", top_items)

    memory_spine = _load_memory_spine(VERIFY_MEMORY_LOOKBACK_DAYS)
    verified_items = top_items
    verify_api_key = _load_mistral_api_key()
    if memory_spine and verify_api_key and _HAS_MISTRAL:
        print(f"Running Mistral editorial verify against {len(memory_spine)} memory-spine entries...")
        try:
            verified_items = _verify_digest_with_mistral(top_items, memory_spine, verify_api_key)
            verified_items = _restore_quota_minimums(top_items, verified_items)
            removed = len(top_items) - len(verified_items)
            print(f"LLM verifier removed {removed} item(s)")
        except Exception as exc:
            print(f"Warning: LLM verify failed ({exc}); using unverified selection")
            verified_items = top_items
    elif memory_spine and not verify_api_key:
        print("Warning: MISTRAL_API_KEY not set; skipping LLM verify step")
    else:
        print("No memory spine entries; skipping LLM verify step")

    _print_top_items_list(f"Top {len(verified_items)} items AFTER LLM verify:", verified_items)

    print("Category counts:")
    for category, count in sorted(_category_counts(verified_items).items()):
        print(f"  {category}: {count}")

    output_data = {
        "date": data.get("date", datetime.now().strftime("%Y-%m-%d")),
        "generated_at": data.get("generated_at", ""),
        "items": verified_items,
        "stats": {
            "total_collected": len(items),
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

    print(f"Filtered digest input saved to: {output_path}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f}KB")


def _category_counts(items: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        cat = _digest_category(item)
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Filter top scored items for digest generation")
    parser.add_argument("input", help="Path to the full collection JSON file")
    parser.add_argument("output", help="Path to save the filtered output JSON")
    parser.add_argument("--top", type=int, default=20, help="Number of top items to keep (default: 20)")
    args = parser.parse_args()

    filter_top_items(args.input, args.output, args.top)


if __name__ == "__main__":
    main()
