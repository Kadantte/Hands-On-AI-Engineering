# Devable Research Agent - Team Workflow

This document explains how the AI Research Agent works for anyone on the team who did not build it. For installation steps and API keys, see [README.md](README.md).

---

## 1. What is this?

The **Devable Research Agent** (also called the AI Research Agent) is an automated daily briefing for AI engineering news. Every weekday morning it collects signals from papers, repos, models, community threads, and web sources, picks the most useful items, and posts a curated digest to Slack.

We built it so the team stays current on what matters for builders (new models, repos, tools, papers, and community signals) without manually checking dozens of sites. The goal is one reliable Slack message each morning, written for engineers who ship products.

---

## 2. What it collects and where from

The agent pulls from **seven sources**. Each source feeds into a shared daily file before filtering.

| Source | What we get | Where it comes from |
|--------|------------|---------------------|
| **arXiv** | Research papers | arXiv API, matched to tracked labs and topics in `entities/` |
| **GitHub** | Releases and trending repos | GitHub API (optional token for higher limits) |
| **HuggingFace** | Trending models and datasets | HuggingFace public API |
| **Hacker News** | Community discussions and Show HN launches | Hacker News via Algolia search |
| **Good AI List** | Curated AI engineering repos | [goodailist.com](https://goodailist.com/repos) (Chip Huyen's list), scraped with Playwright |
| **AINews (smol.ai)** | Daily digest of Reddit, X, and Discord highlights | RSS feed + HTML parsing from [news.smol.ai](https://news.smol.ai) |
| **Tavily** | Web signals and release announcements | Tavily Search API (**currently paused**: free tier credits exhausted) |

**Entity tracking:** We watch 80+ companies, labs, and tools defined in markdown files under `entities/`. Collectors use those lists to decide what to search for, rather than generic keyword scraping.

---

## 3. How the pipeline works

The pipeline runs in three steps: **collect**, **filter**, **deliver**.

### Step 1 - Collect (7:45am Lagos time)

- **Windows Task Scheduler** runs `scripts/run_pipeline.bat`.
- That script calls `scripts/run_pipeline.py`, which runs `skills/tracking-list/scripts/collect.py`.
- All active collectors run in parallel and write one file: **`data_today.json`**.
- A log is appended to **`logs/pipeline.log`**.

### Step 2 - Filter (same run, right after collect)

- `scripts/filter_top_items.py` reads `data_today.json`.
- It selects the best items using:
  - **LLM paper relevance scoring (Mistral):** arXiv papers scored 1-5; low-scoring papers are dropped.
  - **Category diversity quotas:** Ensures mix of papers, repos, web signals, X (via AINews), Hacker News, and Good AI List items (not all from one source).
  - **Frequency cap:** The same GitHub or Good AI List repo will not dominate the digest; repos that appeared twice in the last **30 days** are held back.
  - **Deduplication log:** Items already delivered in the last **7-14 days** (depending on source) are skipped so stories do not repeat.
  - **Mistral editorial verify:** A second pass removes items that duplicate recent deliveries or overlap the same story.
- Output: **`digest_input_today.json`** with up to **20 items**, plus updates to **`logs/delivered_items.json`**.

### Step 3 - Deliver (8:00am Lagos time)

- **OpenClaw cron** runs on the team machine.
- It reads `digest_input_today.json`, editorial rules from `~/.config/morning-ai/editorial_profile.md`, and formatting rules from `scripts/openclaw_digest_prompt.txt`.
- Mistral writes the final digest text and OpenClaw posts it to the team Slack channel automatically.

**Manual runs:** Anyone can run `scripts/run_pipeline.bat` or `python scripts/run_pipeline.py` to collect and filter on demand. Use `scripts/refresh_digest_input.bat` to re-filter without re-collecting.

---

## 4. What the digest looks like

The Slack message uses emoji section headers. **Empty sections are omitted** (if there are no qualifying papers that day, the Papers section does not appear).

Typical structure:

```
📰 AI Engineering Digest - 2026-07-10

📄 Top Papers
- Paper title - short technical description - <https://arxiv.org/abs/1234|link>

📦 Trending Repos
- repo-name - short technical description - <https://github.com/org/repo|link>

💬 What Happened on Reddit
- Thread title - short technical description - <https://www.reddit.com/r/...|link>

🐦 What's Trending on X
- Signal title - short technical description - <https://x.com/user/status/123|link>

🤗 New Model Drops
- Model name - short technical description - <https://huggingface.co/...|link>

🛠 New Tools
- Tool name - short technical description - <https://example.com|link>
```

**Item line rules (important for Slack):**

- Each item is **one line** with three parts: `Title - short description - <url|link>`
- Links must use Slack format `<https://full-url|link>` so Slack does not show large preview cards.
- Bare URLs like `https://example.com` are not allowed in the digest.
- Descriptions should be specific and technical, not marketing fluff.

---

## 5. Model providers

| Provider | Model | Used for |
|----------|-------|----------|
| **Mistral** | Mistral Small (`mistral-small-latest`) | Digest writing (OpenClaw), arXiv paper relevance scoring, editorial verify step in filter |
| **Gemini** | Gemini 2.0 Flash | Planned for native X/Twitter search when Anthropic credits are restored |
| **OpenAI** | GPT-4o-mini | Fallback option in the broader stack |

OpenClaw is configured to use Mistral via an OpenAI-compatible provider for daily digest generation.

---

## 6. Known limitations and pending work

| Area | Status |
|------|--------|
| **X (Twitter)** | Native X collector is off. Anthropic credits for the original approach are exhausted. Gemini has limited X visibility. **AINews covers X indirectly** via smol.ai Twitter sections. |
| **Reddit** | Direct Reddit collection is geo-restricted from Nigeria. **AINews covers Reddit indirectly** via high-activity threads in smol.ai issues. |
| **Tavily** | Free tier credits exhausted. Collector code remains but **web search via Tavily is paused** until credits are renewed. |
| **Good AI List** | Requires **Playwright** and **Chromium** installed on the machine. If missing, Good AI List returns zero items but the rest of the pipeline still runs. |
| **Paths and OS** | Automation is set up for **Windows** (Task Scheduler, batch scripts). Some OpenClaw prompt paths still reference `C:\Users\HP\Documents\MorningAI`. Cross-platform and fully portable paths are pending. |
| **Config location** | API keys and editorial profile live in `~/.config/morning-ai/` (legacy name from the upstream MorningAI project). |

---

## 7. How to set up from scratch

For full setup (Python packages, API keys, Task Scheduler, OpenClaw cron, Slack channel), follow **[README.md](README.md)** in this folder.

Quick checklist for a new team member:

1. Install Python 3.9+ and optional packages (`mistralai`, `feedparser`, `playwright`, `tavily-python`).
2. Copy `.env.example` to `~/.config/morning-ai/.env` and add keys (Mistral required, Tavily when renewed, GitHub token optional).
3. Create `~/.config/morning-ai/editorial_profile.md` with team editorial rules.
4. Configure OpenClaw with Slack and Mistral (see README).
5. Schedule `scripts/run_pipeline.bat` at 7:45am Lagos time.
6. Confirm OpenClaw cron at 8:00am reads `digest_input_today.json`.

**Useful commands:**

```bash
# Full pipeline
python scripts/run_pipeline.py

# Re-filter only (after rule changes)
scripts\refresh_digest_input.bat

# Sync digest prompt to OpenClaw after editing openclaw_digest_prompt.txt
python scripts/update_openclaw_cron.py
```

---

## Questions?

- **Pipeline logs:** `logs/pipeline.log`
- **What was already sent:** `logs/delivered_items.json`
- **Digest input for today:** `digest_input_today.json` (regenerated each run, not committed to git)
- **Developer details:** [README.md](README.md)
