"""AINews RSS collector for devable-research-agent.

Fetches daily AINews issues from smol.ai RSS, then parses each issue
HTML into individual story TrackerItems (Twitter sections, Twitter intro
bullets, and high-activity Reddit threads).
"""

import calendar
import hashlib
import html
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    feedparser = None  # type: ignore[misc, assignment]
    _HAS_FEEDPARSER = False

from . import http
from .schema import TrackerItem, Engagement, CollectionResult, SOURCE_WEB
from .util import log, parse_date

AINEWS_RSS_URL = "https://news.smol.ai/rss.xml"
AINEWS_ENTITY = "AINews (smol.ai)"
AINEWS_LABEL = "AINews"
LOOKBACK_HOURS = 72
DEFAULT_RELEVANCE = 0.55
REDDIT_MIN_ACTIVITY = 100
SUMMARY_LIMIT = 500
TITLE_LIMIT = 200

_H1_SECTION_RE = re.compile(
    r'<h1[^>]*id="([^"]+)"[^>]*>(.*?)(?=<h1[^>]*id=|$)',
    re.IGNORECASE | re.DOTALL,
)
_H2_BLOCK_RE = re.compile(
    r'<h2[^>]*id="([^"]+)"[^>]*>(.*?)</h2>(.*?)(?=<h2[^>]*id=|$)',
    re.IGNORECASE | re.DOTALL,
)
_LI_RE = re.compile(r"<li>(.*?)</li>", re.IGNORECASE | re.DOTALL)
_FIRST_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_X_URL_RE = re.compile(
    r'href="(https?://(?:www\.)?(?:x\.com|twitter\.com)/[^"]+)"',
    re.IGNORECASE,
)
_REDDIT_THREAD_RE = re.compile(
    r'<strong><a href="(https://www\.reddit\.com/[^"]+)">([^<]+)</a></strong>'
    r'\s*\(Activity:\s*(\d+)\)\s*:?\s*(.*?)(?=(?:<strong><a href="https://www\.reddit\.com/)|</li>)',
    re.IGNORECASE | re.DOTALL,
)

_log = lambda msg: log("AINews", msg, tty_only=True)


def _url_id(url: str) -> str:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    return f"WEB-{digest}"


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _truncate(text: str, limit: int = SUMMARY_LIMIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _entry_timestamp(entry: Dict) -> Optional[float]:
    """Return entry publish time as UTC epoch seconds."""
    for attr in ("published_parsed", "updated_parsed"):
        parsed = entry.get(attr)
        if parsed:
            try:
                return float(calendar.timegm(parsed))
            except (TypeError, ValueError, OverflowError):
                continue
    for attr in ("published", "updated"):
        raw = entry.get(attr)
        if raw:
            date_str = parse_date(str(raw))
            if date_str:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                return dt.timestamp()
    return None


def _lookback_window(to_date: str) -> Tuple[float, float]:
    """Return (cutoff_ts, upper_ts) for the 72-hour window ending at to_date."""
    try:
        end = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = end + timedelta(days=1)
    except ValueError:
        end = datetime.now(timezone.utc)
    cutoff = end - timedelta(hours=LOOKBACK_HOURS)
    return cutoff.timestamp(), end.timestamp()


def _fetch_feed(url: str = AINEWS_RSS_URL) -> Tuple[Optional[object], Optional[str]]:
    """Fetch and parse the RSS feed. Returns (feed, error)."""
    if not _HAS_FEEDPARSER:
        return None, "feedparser not installed (pip install feedparser)"

    try:
        body = http.get_text(
            url,
            headers={"Accept": "application/rss+xml, application/xml, text/xml"},
            timeout=20,
            retries=3,
        )
    except http.HTTPError as exc:
        return None, f"fetch failed: {exc}"
    except Exception as exc:
        return None, f"fetch failed: {exc}"

    if not body:
        return None, "empty RSS response"

    try:
        feed = feedparser.parse(body)
    except Exception as exc:
        return None, f"parse failed: {exc}"

    if getattr(feed, "bozo", False) and not getattr(feed, "entries", None):
        bozo_exc = getattr(feed, "bozo_exception", None)
        return None, f"malformed RSS feed: {bozo_exc or 'unknown parse error'}"

    return feed, None


def _fetch_issue_html(issue_url: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch issue page HTML. Returns (article_inner_html, error)."""
    try:
        body = http.get_text(issue_url, timeout=30, retries=3)
    except http.HTTPError as exc:
        return None, f"{issue_url}: fetch failed: {exc}"
    except Exception as exc:
        return None, f"{issue_url}: fetch failed: {exc}"

    if not body:
        return None, f"{issue_url}: empty response"

    match = re.search(
        r'<article class="content-area"[^>]*>(.*?)</article>',
        body,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, f"{issue_url}: content-area not found"

    return match.group(1), None


def _h1_section(article: str, section_id: str) -> Optional[str]:
    """Return inner HTML for a top-level h1 section by id."""
    for hid, content in _H1_SECTION_RE.findall(article):
        if hid.lower() == section_id.lower():
            return content
    return None


def _first_paragraph_summary(chunk: str) -> str:
    match = _FIRST_P_RE.search(chunk)
    if not match:
        return ""
    return _truncate(_strip_html(match.group(1)))


def _make_item(
    title: str,
    summary: str,
    source_url: str,
    issue_date: str,
    *,
    relevance: float = DEFAULT_RELEVANCE,
    engagement: Optional[Engagement] = None,
) -> TrackerItem:
    title = _truncate(title, TITLE_LIMIT)
    summary = _truncate(summary or title, SUMMARY_LIMIT)
    return TrackerItem(
        id=_url_id(source_url),
        title=title,
        summary=summary,
        entity=AINEWS_ENTITY,
        source=SOURCE_WEB,
        source_url=source_url,
        source_label=AINEWS_LABEL,
        date=issue_date,
        date_confidence="high",
        raw_text=summary,
        engagement=engagement or Engagement(),
        relevance=relevance,
    )


def _parse_twitter_h2_sections(
    article: str,
    issue_url: str,
    issue_date: str,
) -> List[TrackerItem]:
    """Each h2 block in the Twitter chapter becomes one TrackerItem."""
    twitter = _h1_section(article, "ai-twitter-recap")
    if not twitter:
        return []

    base = issue_url.rstrip("/")
    items: List[TrackerItem] = []
    for section_id, heading_html, body_html in _H2_BLOCK_RE.findall(twitter):
        title = _strip_html(heading_html)
        if not title:
            continue
        summary = _first_paragraph_summary(body_html) or title
        x_match = _X_URL_RE.search(body_html)
        source_url = x_match.group(1) if x_match else f"{base}#{section_id}"
        items.append(_make_item(title, summary, source_url, issue_date))
    return items


def _parse_twitter_intro_bullets(
    article: str,
    issue_url: str,
    issue_date: str,
) -> List[TrackerItem]:
    """li items under Twitter h1 before the first h2."""
    twitter = _h1_section(article, "ai-twitter-recap")
    if not twitter:
        return []

    intro_match = re.search(r"^(.*?)(?=<h2[^>]*id=)", twitter, re.IGNORECASE | re.DOTALL)
    if not intro_match:
        return []

    base = issue_url.rstrip("/")
    items: List[TrackerItem] = []
    for idx, li_html in enumerate(_LI_RE.findall(intro_match.group(1)), start=1):
        plain = _strip_html(li_html)
        if not plain:
            continue

        x_match = _X_URL_RE.search(li_html)
        source_url = x_match.group(1) if x_match else f"{base}#ai-twitter-recap-bullet-{idx}"
        title = _truncate(plain, TITLE_LIMIT)
        items.append(_make_item(title, plain, source_url, issue_date))
    return items


def _parse_reddit_threads(
    article: str,
    issue_url: str,
    issue_date: str,
) -> List[TrackerItem]:
    """Reddit li items with (Activity: N); only Activity > 100."""
    reddit = _h1_section(article, "ai-reddit-recap")
    if not reddit:
        return []

    items: List[TrackerItem] = []
    seen_urls: set = set()
    for thread_url, title, activity_str, summary_html in _REDDIT_THREAD_RE.findall(reddit):
        activity = int(activity_str)
        if activity <= REDDIT_MIN_ACTIVITY:
            continue
        if thread_url in seen_urls:
            continue
        seen_urls.add(thread_url)

        title = _strip_html(title)
        summary = _truncate(_strip_html(summary_html) or title)
        relevance = min(1.0, DEFAULT_RELEVANCE + (activity / 2000))
        items.append(
            _make_item(
                title,
                summary,
                thread_url,
                issue_date,
                relevance=round(relevance, 2),
                engagement=Engagement(score=activity, total=activity),
            )
        )
    return items


def _parse_issue_stories(
    article: str,
    issue_url: str,
    issue_date: str,
) -> List[TrackerItem]:
    """Extract all story-level TrackerItems from one issue page."""
    items: List[TrackerItem] = []
    items.extend(_parse_twitter_intro_bullets(article, issue_url, issue_date))
    items.extend(_parse_twitter_h2_sections(article, issue_url, issue_date))
    items.extend(_parse_reddit_threads(article, issue_url, issue_date))
    return items


def _recent_issue_entries(entries: List[Dict], to_date: str) -> List[Tuple[str, str]]:
    """Return (issue_url, issue_date) pairs within the 72h lookback window."""
    cutoff_ts, upper_ts = _lookback_window(to_date)
    issues: List[Tuple[str, str]] = []
    for entry in entries:
        link = (entry.get("link") or "").strip()
        if not link:
            continue
        published_ts = _entry_timestamp(entry)
        if published_ts is None:
            continue
        if published_ts < cutoff_ts or published_ts >= upper_ts:
            continue
        issue_date = datetime.fromtimestamp(published_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        issues.append((link, issue_date))
    return issues


def collect(
    from_date: str,
    to_date: str,
    depth: str = "default",
) -> CollectionResult:
    """Collect AINews stories from issues published in the last 72 hours.

    Args:
        from_date: Start of collection window (unused; kept for collector API parity)
        to_date: End date YYYY-MM-DD — 72h lookback ends at start of next day UTC
        depth: Collection depth (unused; single feed)

    Returns:
        CollectionResult with source=\"ainews\". Never raises.
    """
    del from_date, depth  # API parity with other collectors
    result = CollectionResult(source="ainews")
    result.entities_checked = 1

    feed, error = _fetch_feed()
    if error:
        result.errors.append(error)
        _log(error)
        return result

    entries = getattr(feed, "entries", []) or []
    issues = _recent_issue_entries(entries, to_date)
    if not issues:
        _log(f"0 issues in last {LOOKBACK_HOURS}h (from {len(entries)} feed entries)")
        return result

    seen_urls: set = set()
    for issue_url, issue_date in issues:
        article, fetch_error = _fetch_issue_html(issue_url)
        if fetch_error:
            result.errors.append(fetch_error)
            _log(fetch_error)
            continue

        try:
            stories = _parse_issue_stories(article, issue_url, issue_date)
        except Exception as exc:
            result.errors.append(f"{issue_url}: parse error: {exc}")
            _log(f"{issue_url}: parse error: {exc}")
            continue

        added = 0
        for item in stories:
            if item.source_url in seen_urls:
                continue
            seen_urls.add(item.source_url)
            result.items.append(item)
            added += 1

        _log(f"{issue_url}: {added} stories ({len(stories)} parsed)")

    if result.items:
        result.entities_with_updates = 1

    _log(
        f"{len(result.items)} items from {len(issues)} issues "
        f"in last {LOOKBACK_HOURS}h"
    )
    return result
