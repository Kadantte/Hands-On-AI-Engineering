"""Twitter/X collector for devable-research-agent.

Uses the ``twitter-cli`` binary (subprocess) with cookie auth from
``TWITTER_AUTH_TOKEN`` and ``TWITTER_CT0``. Replaces the old Anthropic-based
``x_agent.py`` approach.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import entities
from .env import load_env_file
from .schema import CollectionResult, Engagement, SOURCE_X, TrackerItem
from .util import log, parse_date

SOURCE_LABEL = "Twitter/X"
MIN_VIEWS = 50
TITLE_LIMIT = 120
SEARCH_LIMIT = 20
USER_POSTS_LIMIT = 10
CLI_TIMEOUT = 120

SEARCH_QUERIES = [
    "open source LLM release",
    "new AI agent framework released",
    "open source model weights",
    "new inference engine open source",
]

TEAM_ACCOUNTS = [
    "askalphaxiv",
    "huggingpapers",
    "dair_ai",
    "ArtificialAnlys",
]

_log = lambda msg: log("Twitter", msg, tty_only=True)

_RT_PREFIX_RE = re.compile(r"^\s*RT\s+@", re.IGNORECASE)


def _tweet_id(raw_id: Any) -> str:
    return f"X-{raw_id}"


def _load_twitter_credentials(
    auth_token: Optional[str] = None,
    ct0: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Load Twitter cookie credentials from args, env, or morning-ai config."""
    token = auth_token or os.environ.get("TWITTER_AUTH_TOKEN")
    cookie_ct0 = ct0 or os.environ.get("TWITTER_CT0")

    if token and cookie_ct0:
        return token, cookie_ct0

    merged: Dict[str, str] = {}
    merged.update(load_env_file(str(Path.home() / ".config" / "morning-ai" / ".env")))
    merged.update(load_env_file(".env"))
    merged.update(load_env_file(".env.local"))

    token = token or merged.get("TWITTER_AUTH_TOKEN")
    cookie_ct0 = cookie_ct0 or merged.get("TWITTER_CT0")
    return (token or None), (cookie_ct0 or None)


def _twitter_cli_env(auth_token: str, ct0: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["TWITTER_AUTH_TOKEN"] = auth_token
    env["TWITTER_CT0"] = ct0
    return env


def _run_twitter_cli(args: List[str], env: Dict[str, str]) -> Tuple[Optional[Any], Optional[str]]:
    """Run twitter-cli and parse JSON stdout."""
    binary = shutil.which("twitter")
    if not binary:
        return None, "twitter-cli not found on PATH (install twitter-cli / Agent-Reach)"

    cmd = [binary, *args, "--json"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout after {CLI_TIMEOUT}s: {' '.join(args[:3])}"
    except OSError as exc:
        return None, f"failed to run twitter-cli: {exc}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if not stdout:
        detail = stderr or f"exit code {proc.returncode}"
        return None, f"empty twitter-cli output ({detail})"

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        snippet = stdout[:200].replace("\n", " ")
        return None, f"invalid JSON from twitter-cli: {exc}; snippet={snippet!r}"

    if isinstance(payload, dict) and payload.get("ok") is False:
        err = payload.get("error") or {}
        code = err.get("code") or "unknown"
        message = err.get("message") or str(err)
        return None, f"{code}: {message}"

    return payload, None


def _extract_tweets(payload: Any) -> List[Dict[str, Any]]:
    """Normalize twitter-cli JSON into a list of tweet dicts."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("tweets", "data", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _tweet_text(tweet: Dict[str, Any]) -> str:
    return str(tweet.get("text") or tweet.get("fullText") or "").strip()


def _tweet_author(tweet: Dict[str, Any]) -> str:
    author = tweet.get("author") or {}
    if isinstance(author, dict):
        for key in ("screenName", "screen_name", "username", "userName"):
            handle = author.get(key)
            if handle:
                return str(handle).lstrip("@")
    for key in ("screenName", "screen_name", "username"):
        handle = tweet.get(key)
        if handle:
            return str(handle).lstrip("@")
    return ""


def _tweet_metrics(tweet: Dict[str, Any]) -> Dict[str, int]:
    metrics = tweet.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    views = metrics.get("views") or tweet.get("views") or 0
    likes = metrics.get("likes") or tweet.get("likes") or 0
    retweets = metrics.get("retweets") or tweet.get("retweets") or 0
    replies = metrics.get("replies") or tweet.get("replies") or 0
    return {
        "views": int(views or 0),
        "likes": int(likes or 0),
        "retweets": int(retweets or 0),
        "replies": int(replies or 0),
    }


def _tweet_created_iso(tweet: Dict[str, Any]) -> str:
    for key in ("createdAtISO", "created_at", "createdAt", "date"):
        value = tweet.get(key)
        if value:
            return str(value)
    return ""


def _tweet_date(iso_value: str, from_date: str, to_date: str) -> Optional[str]:
    if not iso_value:
        return None
    try:
        normalized = iso_value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
    except ValueError:
        date_str = parse_date(iso_value)
        if not date_str:
            return None
    if from_date and date_str < from_date:
        return None
    if to_date and date_str > to_date:
        return None
    return date_str


def _is_retweet(tweet: Dict[str, Any], text: str) -> bool:
    if tweet.get("isRetweet") or tweet.get("retweeted"):
        return True
    return bool(_RT_PREFIX_RE.match(text))


def _is_english(text: str) -> bool:
    if not text.strip():
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    non_latin = sum(1 for c in letters if ord(c) > 127)
    return (non_latin / len(letters)) < 0.2


def _compute_relevance(views: int, likes: int) -> float:
    view_score = min(1.0, views / 10_000)
    like_score = min(1.0, likes / 500)
    relevance = min(1.0, view_score * 0.45 + like_score * 0.45 + 0.1)
    return round(relevance, 2)


def _tweet_url(tweet: Dict[str, Any], screen_name: str) -> str:
    tweet_id = tweet.get("id") or tweet.get("id_str")
    if tweet_id and screen_name:
        return f"https://x.com/{screen_name}/status/{tweet_id}"
    urls = tweet.get("urls") or []
    if isinstance(urls, list):
        for entry in urls:
            if isinstance(entry, str) and entry.startswith("http"):
                return entry
            if isinstance(entry, dict):
                expanded = entry.get("expanded_url") or entry.get("url")
                if expanded:
                    return str(expanded)
    return ""


def _kol_handles(x_handles_kol: Optional[Dict[str, List[str]]] = None) -> List[str]:
    registry = x_handles_kol if x_handles_kol is not None else entities.X_HANDLES_KOL
    handles: List[str] = []
    seen: Set[str] = set()
    for handle_list in registry.values():
        for handle in handle_list:
            normalized = str(handle).lstrip("@").lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                handles.append(normalized)
    for handle in TEAM_ACCOUNTS:
        normalized = handle.lstrip("@").lower()
        if normalized not in seen:
            seen.add(normalized)
            handles.append(normalized)
    return handles


def _entity_for_handle(handle: str, x_handles_kol: Dict[str, List[str]]) -> str:
    needle = handle.lower()
    for entity_name, handles in x_handles_kol.items():
        for raw in handles:
            if str(raw).lstrip("@").lower() == needle:
                return entity_name
    if needle in {h.lower() for h in TEAM_ACCOUNTS}:
        return handle
    return handle


def _tweet_to_item(
    tweet: Dict[str, Any],
    *,
    entity: str,
    from_date: str,
    to_date: str,
) -> Optional[TrackerItem]:
    text = _tweet_text(tweet)
    if not text:
        return None
    if _is_retweet(tweet, text):
        return None
    if not _is_english(text):
        return None

    metrics = _tweet_metrics(tweet)
    if metrics["views"] < MIN_VIEWS:
        return None

    created_iso = _tweet_created_iso(tweet)
    date_str = _tweet_date(created_iso, from_date, to_date)
    if not date_str:
        return None

    screen_name = _tweet_author(tweet)
    source_url = _tweet_url(tweet, screen_name)
    if not source_url:
        return None

    title = text if len(text) <= TITLE_LIMIT else text[: TITLE_LIMIT - 3].rstrip() + "..."
    engagement = Engagement(
        likes=metrics["likes"],
        reposts=metrics["retweets"],
        replies=metrics["replies"],
        views=metrics["views"],
        total=metrics["likes"] + metrics["retweets"] + metrics["replies"],
    )

    return TrackerItem(
        id=_tweet_id(tweet.get("id") or tweet.get("id_str") or source_url),
        title=title,
        summary=text,
        entity=entity,
        content_type="",
        source=SOURCE_X,
        source_url=source_url,
        source_label=SOURCE_LABEL,
        date=date_str,
        date_confidence="high" if created_iso else "med",
        raw_text=text,
        engagement=engagement,
        relevance=_compute_relevance(metrics["views"], metrics["likes"]),
    )


def _search_tweets(
    query: str,
    from_date: str,
    to_date: str,
    env: Dict[str, str],
) -> Tuple[List[TrackerItem], Optional[str]]:
    args = [
        "search",
        query,
        "-t",
        "latest",
        "-n",
        str(SEARCH_LIMIT),
        "--lang",
        "en",
        "--exclude",
        "retweets",
    ]
    if from_date:
        args.extend(["--since", from_date])
    if to_date:
        args.extend(["--until", to_date])

    payload, error = _run_twitter_cli(args, env)
    if error:
        return [], f"search {query!r}: {error}"

    items: List[TrackerItem] = []
    for tweet in _extract_tweets(payload):
        item = _tweet_to_item(
            tweet,
            entity="Twitter Search",
            from_date=from_date,
            to_date=to_date,
        )
        if item:
            items.append(item)
    _log(f"Search {query!r}: {len(items)} items")
    return items, None


def _user_posts(
    handle: str,
    entity: str,
    from_date: str,
    to_date: str,
    env: Dict[str, str],
) -> Tuple[List[TrackerItem], Optional[str]]:
    args = ["user-posts", handle.lstrip("@"), "-n", str(USER_POSTS_LIMIT)]
    payload, error = _run_twitter_cli(args, env)
    if error:
        return [], f"user-posts @{handle}: {error}"

    items: List[TrackerItem] = []
    for tweet in _extract_tweets(payload):
        item = _tweet_to_item(
            tweet,
            entity=entity,
            from_date=from_date,
            to_date=to_date,
        )
        if item:
            items.append(item)
    _log(f"@{handle}: {len(items)} items")
    return items, None


def collect(
    _github_sources: Any,
    from_date: str,
    to_date: str,
    depth: str = "default",
    *,
    auth_token: Optional[str] = None,
    ct0: Optional[str] = None,
    x_handles_kol: Optional[Dict[str, List[str]]] = None,
) -> CollectionResult:
    """Collect tweets via twitter-cli search + KOL/team timelines.

    Args:
        _github_sources: Unused (kept for collect.py / test API parity).
        from_date: Window start YYYY-MM-DD.
        to_date: Window end YYYY-MM-DD.
        depth: quick | default | deep (reserved; limits fixed for now).
        auth_token: Optional TWITTER_AUTH_TOKEN override.
        ct0: Optional TWITTER_CT0 override.
        x_handles_kol: Optional KOL handle registry override.
    """
    del _github_sources, depth  # depth reserved for future tuning

    result = CollectionResult(source=SOURCE_X)
    token, cookie_ct0 = _load_twitter_credentials(auth_token, ct0)
    if not token or not cookie_ct0:
        result.errors.append(
            "TWITTER_AUTH_TOKEN and TWITTER_CT0 not configured "
            "(set in ~/.config/morning-ai/.env or environment)"
        )
        return result

    env = _twitter_cli_env(token, cookie_ct0)
    kol_registry = x_handles_kol if x_handles_kol is not None else entities.X_HANDLES_KOL
    handles = _kol_handles(kol_registry)
    result.entities_checked = len(SEARCH_QUERIES) + len(handles)

    all_items: List[TrackerItem] = []
    seen_urls: Set[str] = set()
    entities_with_updates: Set[str] = set()
    tasks: List[Tuple[str, str, Optional[str]]] = []
    for query in SEARCH_QUERIES:
        tasks.append(("search", query, None))
    for handle in handles:
        tasks.append(("user", handle, _entity_for_handle(handle, kol_registry)))

    max_workers = min(8, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for kind, key, entity in tasks:
            if kind == "search":
                futures[pool.submit(_search_tweets, key, from_date, to_date, env)] = (
                    "search",
                    key,
                )
            else:
                futures[pool.submit(_user_posts, key, entity or key, from_date, to_date, env)] = (
                    "user",
                    key,
                )

        for future in as_completed(futures):
            kind, key = futures[future]
            try:
                items, error = future.result()
            except Exception as exc:
                result.errors.append(f"{kind} {key}: {exc}")
                continue
            if error:
                result.errors.append(error)
                continue
            for item in items:
                if item.source_url in seen_urls:
                    continue
                seen_urls.add(item.source_url)
                all_items.append(item)
                entities_with_updates.add(item.entity)

    result.items = all_items
    result.entities_with_updates = len(entities_with_updates)
    _log(f"Collected {len(all_items)} Twitter items ({len(result.errors)} errors)")
    return result
