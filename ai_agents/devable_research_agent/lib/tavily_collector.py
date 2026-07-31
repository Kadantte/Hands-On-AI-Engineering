"""Tavily web search collector for devable-research-agent.

Uses the Tavily Search API to find general web signals:

1. Entity ``web_queries`` from the entity registry (site: queries, etc.)
2. Fallback entity news queries for tracked entities without explicit web queries
3. High-signal release/announcement discovery queries

X/Twitter handle searches were removed — AINews RSS covers X signals.

Requires ``tavily-python`` and ``TAVILY_API_KEY`` in config or environment.
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

try:
    from tavily import TavilyClient
    _HAS_TAVILY = True
except ImportError:
    TavilyClient = None  # type: ignore[misc, assignment]
    _HAS_TAVILY = False

from .schema import TrackerItem, Engagement, CollectionResult, SOURCE_WEB
from .util import log, parse_date

TAVILY_SEARCH_DEPTH = {"quick": "basic", "default": "advanced", "deep": "advanced"}
MAX_RESULTS = {"quick": 5, "default": 8, "deep": 10}
DEPTH_WEB_ENTITIES = {"quick": 15, "default": 35, "deep": 60}
MAX_WORKERS = 4

# Release-focused AI engineering discovery queries (not news commentary)
HIGH_SIGNAL_WEB_QUERIES = [
    "site:news.smol.ai AI engineering signals",
    "new open source LLM model released this week site:huggingface.co",
    "new AI agent framework released github this week",
    "open source model weights released arxiv 2026",
    "new LLM inference engine released open source",
    "new open source benchmark released AI engineering",
    "AI tool release announcement github 2026",
    "open weight model released this week",
    "new multimodal model open source released",
    "new open source coding agent released",
    "new RAG framework released open source",
]
HIGH_SIGNAL_ENTITY = "AI Engineering Web"

# Skip low-signal results (mirrors HN noise filtering spirit)
NOISE_PATTERNS = [
    r"(?i)\b(opinion|hot take|rant|thread)\b",
    r"(?i)\b(hiring|we're hiring|job opening)\b",
    r"(?i)\bjust shipped a (bug|patch) fix\b",
]

_log = lambda msg: log("Tavily", msg, tty_only=True)


def _url_id(url: str) -> str:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    return f"WEB-{digest}"


def _is_noise(title: str, content: str = "") -> bool:
    text = f"{title} {content}"
    for pat in NOISE_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def _compute_relevance(tavily_score: Optional[float], rank: int) -> float:
    """Blend Tavily relevance score with result rank position."""
    rank_score = max(0.3, 1.0 - (rank * 0.03))
    score_part = min(1.0, float(tavily_score or 0.5))
    relevance = min(1.0, rank_score * 0.45 + score_part * 0.45 + 0.1)
    return round(relevance, 2)


def _build_entity_news_query(entity_name: str) -> str:
    return (
        f'"{entity_name}" AI '
        "(model release OR framework OR agent OR benchmark OR open source)"
    )


def build_web_tasks(
    web_queries: Dict[str, List[str]],
    entity_names: List[str],
    depth: str = "default",
) -> List[Tuple[str, str]]:
    """Build the Tavily web query task list (entity, query) pairs."""
    web_entity_cap = DEPTH_WEB_ENTITIES.get(depth, DEPTH_WEB_ENTITIES["default"])
    web_tasks: List[Tuple[str, str]] = []

    for entity_name in list(web_queries.keys())[:web_entity_cap]:
        for query in web_queries.get(entity_name, []):
            if query.strip():
                web_tasks.append((entity_name, query.strip()))

    for entity_name in entity_names[:web_entity_cap]:
        if entity_name not in web_queries:
            web_tasks.append((entity_name, _build_entity_news_query(entity_name)))

    for query in HIGH_SIGNAL_WEB_QUERIES:
        web_tasks.append((HIGH_SIGNAL_ENTITY, query))

    return web_tasks


def search(
    client: Any,
    query: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    *,
    topic: str = "news",
    include_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run a single Tavily search and return normalized hit dicts."""
    max_results = MAX_RESULTS.get(depth, MAX_RESULTS["default"])
    search_depth = TAVILY_SEARCH_DEPTH.get(depth, TAVILY_SEARCH_DEPTH["default"])

    try:
        response = client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            start_date=from_date,
            end_date=to_date,
            max_results=max_results,
            include_domains=include_domains,
            timeout=60,
        )
    except Exception as e:
        _log(f"Search failed for '{query[:80]}': {e}")
        return []

    return response.get("results", []) or []


def _hit_to_item(
    hit: Dict[str, Any],
    entity: str,
    rank: int,
    *,
    source_label: str,
) -> Optional[TrackerItem]:
    url = (hit.get("url") or "").strip()
    title = (hit.get("title") or "").strip()
    content = (hit.get("content") or "").strip()

    if not url or not title:
        return None

    if _is_noise(title, content):
        return None

    date_str = parse_date(hit.get("published_date"))
    summary = content[:300] if content else title[:300]
    tavily_score = hit.get("score")

    return TrackerItem(
        id=_url_id(url),
        title=title,
        summary=summary,
        entity=entity,
        source=SOURCE_WEB,
        source_url=url,
        source_label=source_label,
        date=date_str,
        date_confidence="high" if date_str else "med",
        raw_text=content or title,
        engagement=Engagement(),
        relevance=_compute_relevance(tavily_score, rank),
    )


def _search_web_query(
    client: Any,
    entity: str,
    query: str,
    from_date: str,
    to_date: str,
    depth: str,
) -> List[TrackerItem]:
    hits = search(client, query, from_date, to_date, depth, topic="news")

    items: List[TrackerItem] = []
    for i, hit in enumerate(hits):
        item = _hit_to_item(
            hit,
            entity,
            i,
            source_label="Tavily",
        )
        if item:
            items.append(item)

    _log(f"Web '{query[:60]}' ({entity}): {len(items)} items")
    return items


def collect(
    x_handles_official: Dict[str, List[str]],
    x_handles_key_people: Dict[str, List[str]],
    web_queries: Dict[str, List[str]],
    from_date: str,
    to_date: str,
    api_key: Optional[str] = None,
    depth: str = "default",
    x_handles_kol: Optional[Dict[str, List[str]]] = None,
) -> CollectionResult:
    """Collect AI news signals via Tavily Search.

    Args:
        x_handles_official: Unused (kept for collect.py API compatibility)
        x_handles_key_people: Unused (kept for collect.py API compatibility)
        web_queries: Entity -> web search queries (site: queries, etc.)
        from_date: Start date YYYY-MM-DD
        to_date: End date YYYY-MM-DD
        api_key: Tavily API key
        depth: quick | default | deep
        x_handles_kol: Unused (kept for collect.py API compatibility)

    Returns:
        CollectionResult with source=web
    """
    result = CollectionResult(source=SOURCE_WEB)

    if not _HAS_TAVILY:
        result.errors.append("tavily-python not installed — run: pip install tavily-python")
        return result

    if not api_key:
        result.errors.append("TAVILY_API_KEY not configured")
        return result

    client = TavilyClient(api_key=api_key)
    all_items: List[TrackerItem] = []
    seen_urls: set = set()

    tracked_entities = sorted(
        set(web_queries)
        | set(x_handles_official)
        | set(x_handles_key_people)
        | (set(x_handles_kol) if x_handles_kol else set())
    )
    result.entities_checked = len(tracked_entities)

    web_tasks = build_web_tasks(web_queries, tracked_entities, depth)

    entities_with_updates: set = set()
    if web_tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_search_web_query, client, entity, query, from_date, to_date, depth): (entity, query)
                for entity, query in web_tasks
            }
            for future in as_completed(futures):
                entity, query = futures[future]
                try:
                    items = future.result()
                except Exception as e:
                    result.errors.append(f"Web search '{query[:40]}': {e}")
                    continue
                for item in items:
                    if item.source_url in seen_urls:
                        continue
                    seen_urls.add(item.source_url)
                    all_items.append(item)
                    entities_with_updates.add(entity)

    result.entities_with_updates = len(entities_with_updates)
    result.items = all_items
    _log(f"Collected {len(all_items)} Tavily items ({len(web_tasks)} web queries)")
    return result
