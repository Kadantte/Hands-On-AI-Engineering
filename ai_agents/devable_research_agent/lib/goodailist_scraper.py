"""Good AI List scraper for devable-research-agent.



Scrapes trending open-source AI repos from https://goodailist.com/repos using

Playwright (JavaScript-rendered table). Extracts repo name, description,

GitHub URL, star counts, and weekly star momentum.



Requires ``playwright`` and a Chromium install (``playwright install chromium``).

"""



import hashlib

import math

import os

import re

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import Any, Dict, List, Optional



try:

    from playwright.sync_api import sync_playwright

    _HAS_PLAYWRIGHT = True

except ImportError:

    sync_playwright = None  # type: ignore[misc, assignment]

    _HAS_PLAYWRIGHT = False



from .schema import TrackerItem, Engagement, CollectionResult, SOURCE_REPO

from .util import log, parse_date



GOOD_AI_LIST_URL = "https://goodailist.com/repos"

DEBUG_HTML_PATH = "goodailist_debug.html"

AI_ENGINEERING_CATEGORY = "AI Engineering"

DEPTH_LIMITS = {"quick": 25, "default": 50, "deep": 100}

PAGE_TIMEOUT_MS = 90_000

TABLE_READY_SELECTOR = "#repos-table tbody tr td.repo-cell a[href*='github.com']"

TZ_CN = timezone(timedelta(hours=8))



_EXTRACT_ROWS_JS = """

() => {

    const parseIntSafe = (text) => {

        const n = parseInt(String(text || '').replace(/,/g, ''), 10);

        return Number.isFinite(n) ? n : 0;

    };

    const parseMomentum = (text) => {

        const lines = String(text || '').split('\\n').map((s) => s.trim()).filter(Boolean);

        const deltaRaw = lines[0] || '';

        const pctRaw = lines[1] || '';

        const delta = parseIntSafe(deltaRaw.replace(/[^0-9-]/g, ''));

        const pct = parseFloat(String(pctRaw).replace('%', '')) || 0;

        return { delta, pct };

    };



    const rows = [];

    for (const tr of document.querySelectorAll('#repos-table tbody tr')) {

        const repoCell = tr.querySelector('td.repo-cell');

        const link = repoCell?.querySelector('a[href*="github.com"]');

        if (!link) continue;



        const owner = (link.querySelector('.repo-owner')?.textContent || '').trim();

        const name = (link.querySelector('.repo-name')?.textContent || '').trim();

        const repoName = (link.getAttribute('title') || '').trim()

            || (owner && name ? `${owner}/${name}` : '');

        if (!repoName.includes('/')) continue;



        const rank = parseIntSafe(tr.querySelector('td.dim')?.textContent || '');



        const starsCell = repoCell?.nextElementSibling;

        const stars = parseIntSafe(starsCell?.textContent || '');



        const day1Cell = starsCell?.nextElementSibling;

        const day1 = parseMomentum(day1Cell?.innerText || '');

        const day7Cell = day1Cell?.nextElementSibling;

        const weekly = parseMomentum(day7Cell?.innerText || '');



        const forksCell = day7Cell?.nextElementSibling;

        const forks = parseIntSafe(forksCell?.textContent || '');



        const description = (tr.querySelector('.desc-cell')?.textContent || '').trim();

        const category = (tr.querySelector('.category-link')?.textContent || '').trim();

        const subcategory = [...tr.querySelectorAll('.subcat-link')]

            .map((el) => el.textContent.trim())

            .filter(Boolean)

            .join(', ');



        const isoDates = [...tr.querySelectorAll('td.dim')]

            .map((td) => td.textContent.trim())

            .filter((value) => /^\\d{4}-\\d{2}-\\d{2}$/.test(value));

        const created = isoDates[0] || '';

        const updated = isoDates[1] || isoDates[0] || '';



        rows.push({

            repo_name: repoName,

            url: link.href,

            rank,

            stars,

            daily_stars: day1.delta,

            daily_pct: day1.pct,

            weekly_stars: weekly.delta,

            weekly_pct: weekly.pct,

            forks,

            description,

            category,

            subcategory,

            created,

            updated,

        });

    }

    return rows;

}

"""



_log = lambda msg: log("GoodAIList", msg, tty_only=True)





def _pipeline_log_path() -> Path:

    root = Path(os.environ.get("DEVABLE_PROJECT_ROOT", ".")).resolve()

    return root / "logs" / "pipeline.log"





def _pipeline_log(message: str) -> None:

    path = _pipeline_log_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S")

    with path.open("a", encoding="utf-8") as handle:

        handle.write(f"[{timestamp}] {message}\n")





def _repo_id(repo_name: str) -> str:

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", repo_name).strip("-").lower()

    if slug:

        return f"REPO-{slug}"

    digest = hashlib.md5(repo_name.encode("utf-8")).hexdigest()[:12]

    return f"REPO-{digest}"





def _apply_ai_engineering_new_only_filters(page) -> None:

    """Filter Good AI List to AI Engineering category and NEW repos only."""

    page.goto(GOOD_AI_LIST_URL, wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)

    page.wait_for_selector(TABLE_READY_SELECTOR, timeout=PAGE_TIMEOUT_MS)



    page.locator("#category-filter-btn").click()

    page.locator("#category-filter-menu").wait_for(state="visible", timeout=PAGE_TIMEOUT_MS)



    for checkbox in page.locator("#category-filter-menu input[type=checkbox]").all():

        checkbox_id = checkbox.get_attribute("id") or ""

        want_checked = checkbox_id == f"cat-{AI_ENGINEERING_CATEGORY}"

        if checkbox.is_checked() != want_checked:

            checkbox.set_checked(want_checked)



    page.locator("#category-filter-btn").click()



    new_only = page.locator("#repos-new-only")

    if not new_only.is_checked():

        new_only.check()



    page.wait_for_load_state("networkidle")

    page.wait_for_selector(TABLE_READY_SELECTOR, timeout=PAGE_TIMEOUT_MS)





def _build_entity_lookup(

    github_sources: Dict[str, Dict[str, Any]],

) -> tuple:

    org_to_entity: Dict[str, str] = {}

    repo_to_entity: Dict[str, str] = {}

    for entity_name, sources in github_sources.items():

        for org in sources.get("orgs", []):

            org_to_entity[org.lower()] = entity_name

        for repo in sources.get("repos", []):

            repo_to_entity[repo.lower()] = entity_name

    return org_to_entity, repo_to_entity





def _match_entity(

    repo_name: str,

    org_to_entity: Dict[str, str],

    repo_to_entity: Dict[str, str],

) -> Optional[str]:

    key = repo_name.lower()

    if key in repo_to_entity:

        return repo_to_entity[key]

    owner = key.split("/")[0] if "/" in key else ""

    return org_to_entity.get(owner)





def _compute_relevance(rank: int, weekly_stars: int, weekly_pct: float, category: str) -> float:

    rank_score = max(0.3, 1.0 - max(0, rank - 1) * 0.008)

    momentum = min(1.0, math.log1p(max(0, weekly_stars)) / 9.0 + max(0.0, weekly_pct) / 250.0)

    relevance = min(1.0, rank_score * 0.45 + momentum * 0.45 + 0.1)

    if "ai engineering" in (category or "").lower():

        relevance = min(1.0, relevance + 0.05)

    return round(relevance, 2)





def dump_page_html(path: str = DEBUG_HTML_PATH) -> int:

    """Navigate to Good AI List and save rendered HTML for DOM debugging."""

    if not _HAS_PLAYWRIGHT:

        raise RuntimeError("playwright not installed — run: pip install playwright")



    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        try:

            page = browser.new_page()

            _apply_ai_engineering_new_only_filters(page)

            html = page.content()

            with open(path, "w", encoding="utf-8") as handle:

                handle.write(html)

        finally:

            browser.close()

    return len(html)





def _scrape_rows_once(depth: str, debug_html_path: Optional[str]) -> List[Dict[str, Any]]:

    limit = DEPTH_LIMITS.get(depth, DEPTH_LIMITS["default"])

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        try:

            page = browser.new_page()

            _apply_ai_engineering_new_only_filters(page)

            if debug_html_path:

                with open(debug_html_path, "w", encoding="utf-8") as handle:

                    handle.write(page.content())

            rows = page.evaluate(_EXTRACT_ROWS_JS)

        finally:

            browser.close()



    if not isinstance(rows, list):

        return []



    filtered = [

        row for row in rows

        if (row.get("category") or "").strip().lower() == "ai engineering"

    ]

    _log(

        f"Scraped {len(rows)} rows after filters; "

        f"{len(filtered)} AI Engineering repos"

    )

    return filtered[:limit]





def scrape_rows(depth: str = "default", debug_html_path: Optional[str] = None) -> List[Dict[str, Any]]:

    """Scrape AI Engineering NEW repos from Good AI List (with one retry)."""

    if not _HAS_PLAYWRIGHT:

        raise RuntimeError("playwright not installed — run: pip install playwright")



    last_error: Optional[Exception] = None

    for attempt in range(2):

        try:

            rows = _scrape_rows_once(depth, debug_html_path)

            if rows:

                return rows

            if attempt == 0:

                _log("Good AI List returned 0 items on first attempt — retrying once")

                continue

            _pipeline_log(

                "Good AI List returned 0 items - check Playwright and Chromium installation"

            )

            return []

        except Exception as exc:

            last_error = exc

            if attempt == 0:

                _log(f"Good AI List scrape failed (attempt 1): {exc} — retrying once")

                continue

            raise exc



    if last_error:

        raise last_error

    _pipeline_log(

        "Good AI List returned 0 items - check Playwright and Chromium installation"

    )

    return []





def _row_to_item(

    row: Dict[str, Any],

    entity: str,

    to_date: str,

) -> Optional[TrackerItem]:

    repo_name = (row.get("repo_name") or "").strip()

    url = (row.get("url") or "").strip()

    description = (row.get("description") or "").strip()

    if not repo_name or not url:

        return None



    updated = parse_date(row.get("updated")) or to_date

    weekly_stars = int(row.get("weekly_stars") or 0)

    weekly_pct = float(row.get("weekly_pct") or 0)

    stars = int(row.get("stars") or 0)

    forks = int(row.get("forks") or 0)

    rank = int(row.get("rank") or 0)

    category = row.get("category") or ""



    summary = description[:300] if description else f"Trending repo with {stars:,} stars"

    if weekly_stars:

        summary = f"{summary} (+{weekly_stars:,} stars this week, {weekly_pct:+.1f}%)".strip()



    return TrackerItem(

        id=_repo_id(repo_name),

        title=repo_name,

        summary=summary[:300],

        entity=entity,

        source=SOURCE_REPO,

        source_url=url,

        source_label="Good AI List",

        date=updated,

        date_confidence="high" if updated else "med",

        raw_text=description,

        engagement=Engagement(stars=stars, forks=forks, total=max(0, weekly_stars)),

        relevance=_compute_relevance(rank, weekly_stars, weekly_pct, category),

    )





def collect(

    github_sources: Dict[str, Dict[str, Any]],

    from_date: str,

    to_date: str,

    depth: str = "default",

) -> CollectionResult:

    """Scrape NEW AI Engineering repos from Good AI List.



    Args:

        github_sources: Entity registry used to match scraped repos to tracked entities

        from_date: Start date YYYY-MM-DD (unused for NEW-only scrape)

        to_date: End date YYYY-MM-DD

        depth: quick | default | deep — controls how many top repos to keep



    Returns:

        CollectionResult with source=repo. Never raises.

    """

    result = CollectionResult(source=SOURCE_REPO)

    result.entities_checked = 1



    if not _HAS_PLAYWRIGHT:

        result.errors.append("playwright not installed — run: pip install playwright")

        _pipeline_log(

            "Good AI List returned 0 items - check Playwright and Chromium installation"

        )

        return result



    try:

        rows = scrape_rows(depth)

    except Exception as e:

        result.errors.append(f"Good AI List scrape failed: {e}")

        _log(f"Scrape failed: {e}")

        _pipeline_log(

            "Good AI List returned 0 items - check Playwright and Chromium installation"

        )

        return result



    if not rows:

        result.errors.append("Good AI List scrape returned no AI Engineering NEW repos")

        _pipeline_log(

            "Good AI List returned 0 items - check Playwright and Chromium installation"

        )

        return result



    org_to_entity, repo_to_entity = _build_entity_lookup(github_sources)

    all_items: List[TrackerItem] = []

    seen_urls: set = set()

    matched_entities: set = set()



    for row in rows:

        repo_name = row.get("repo_name", "")

        url = row.get("url", "")

        if url in seen_urls:

            continue



        matched = _match_entity(repo_name, org_to_entity, repo_to_entity)

        entity = matched or "Good AI List Trending"



        item = _row_to_item(row, entity, to_date)

        if not item:

            continue



        seen_urls.add(url)

        all_items.append(item)

        if matched:

            matched_entities.add(matched)



    result.items = all_items

    result.entities_with_updates = 1 if all_items else 0

    _log(

        f"Collected {len(all_items)} AI Engineering NEW repos from Good AI List "

        f"({len(matched_entities)} matched tracked entities)"

    )

    return result

