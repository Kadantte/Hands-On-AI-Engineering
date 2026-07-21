# Devable Research Agent - Team Workflow

This document explains how the AI Research Agent works for anyone on the team who did not build it. For installation steps and API keys, see [README.md](README.md).

---

## 1. What is this?

The **Devable Research Agent** (also called the AI Research Agent) is an automated daily briefing for AI engineering news. Every weekday morning it collects signals from papers, repos, models, community threads, X, and web sources, picks the most useful items, and posts a curated digest to Slack.

We built it so the team stays current on what matters for builders (new models, repos, tools, papers, and community signals) without manually checking dozens of sites. The goal is one reliable Slack message each morning, written for engineers who ship products.

---

## 2. What it collects and where from

The agent pulls from **eight sources**. Each source feeds into a shared daily file before filtering. Tavily is still in the table but paused when credits run out.

| Source | What we get | Where it comes from |
|--------|------------|---------------------|
| **arXiv** | Research papers | arXiv API, matched to tracked labs and topics in `entities/` |
| **GitHub** | Releases and trending repos | GitHub API (optional token for higher limits) |
| **HuggingFace** | Trending models and datasets | HuggingFace public API |
| **Hacker News** | Community discussions and Show HN launches | Hacker News via Algolia search |
| **Good AI List** | Curated AI engineering repos | [goodailist.com](https://goodailist.com/repos) (Chip Huyen's list), scraped with Playwright |
| **AINews (smol.ai)** | Daily digest of Reddit, X, and Discord highlights | RSS feed + HTML parsing from [news.smol.ai](https://news.smol.ai) |
| **Twitter/X** | KOL posts, team accounts, and search hits for LLM releases, agent frameworks, model weights, and inference engines | `twitter-cli` via `lib/twitter_collector.py` (subprocess). Auth: `TWITTER_AUTH_TOKEN` + `TWITTER_CT0` in `~/.config/morning-ai/.env`. Watches `@askalphaxiv`, `@huggingpapers`, `@dair_ai`, `@ArtificialAnlys` plus fixed search queries. Replaces the old `x_agent.py` (Anthropic API), which is deprecated and not used. |
| **Tavily** | Web signals and release announcements | Tavily Search API (**currently paused** when free tier credits are exhausted) |

**Entity tracking:** We watch 80+ companies, labs, and tools defined in markdown files under `entities/`. Collectors use those lists to decide what to search for, rather than generic keyword scraping.

---

## 3. How the pipeline works

The pipeline runs in three steps: **collect**, **filter**, **deliver**.

### Step 0 - OpenClaw gateway (7:30am Lagos time)

- **Windows Task Scheduler** runs `scripts/start_openclaw_gateway.bat` (or the registered OpenClaw Gateway task) so the gateway is up before the pipeline and cron.
- OpenClaw version: **2026.7.1**.
- Telegram is **disabled** in `~/.openclaw/openclaw.json`; Slack is the only delivery channel.

### Step 1 - Collect (7:45am Lagos time)

- **Windows Task Scheduler** runs `scripts/run_pipeline.bat`.
- That script calls `scripts/run_pipeline.py`, which runs `skills/tracking-list/scripts/collect.py`.
- All active collectors run in parallel and write one file: **`data_today.json`**.
- A log is appended to **`logs/pipeline.log`**.

### Step 2 - Filter (same run, right after collect)

- `scripts/filter_top_items.py` reads `data_today.json`.
- It selects the best items using:
  - **LLM paper relevance scoring (Mistral):** arXiv papers scored 1-5; low-scoring papers are dropped. Scores are cached and written to **`logs/paper_scores_today.json`**.
  - **Pre-written digest descriptions:** Mistral generates a short one-line `digest_description` per item during filter so OpenClaw has consistent copy to work from.
  - **Category diversity quotas:** Rebalanced min/max per bucket (papers, GitHub, web, AINews, X, Hacker News, HuggingFace, Good AI List) so the digest is not dominated by one source.
  - **Frequency cap:** The same GitHub or Good AI List repo will not dominate the digest; repos that appeared twice in the last **30 days** are held back.
  - **Deduplication log:** Items already delivered are skipped. GitHub **releases** dedupe by repo + semver. **Shorter lookbacks (3 days)** apply to GitHub Trending, AINews, and HN Show/Launch items so fresh stories can surface without repeating older ones.
  - **Mistral editorial verify:** A second pass removes items that duplicate recent deliveries or overlap the same story.
  - **Rejected items log:** Dropped items are written to **`logs/rejected_items_today.json`** for debugging.
- Output: **`digest_input_today.json`** with up to **20 items**, plus updates to **`logs/delivered_items.json`**.
- **Debug:** Run filter with `--debug` for a per-stage breakdown (collection counts, dedup, quotas, verify).

### Step 3 - Deliver (8:00am Lagos time)

- **OpenClaw cron** runs on the team machine (job: AI Research Agent).
- Model: **`mistral-openai/mistral-small-latest`**.
- Cron delivery mode is **`none`** (not `announce`) so OpenClaw does not auto-post the agent's final reply.
- The cron job allows **`read`** and **`message`** tools only (`toolsAllow: ["read", "message"]`).
- The agent reads `digest_input_today.json`, editorial rules from `~/.config/morning-ai/editorial_profile.md`, and formatting rules from `scripts/openclaw_digest_prompt.txt`.
- The agent posts to Slack channel **`C0BH9MD221H`** via the OpenClaw **`message` tool** in two steps:
  1. **Top-level message:** title only, e.g. `📰 AI Engineering Digest - 2026-07-17`
  2. **Thread reply:** all section content (papers, repos, X, etc.) using `threadId` from step 1's returned `messageId`
- The agent's final assistant reply is a brief internal confirmation only (not posted to Slack).
- Slack config in `~/.openclaw/openclaw.json`: **`unfurlLinks: false`** and **`unfurlMedia: false`** so bot messages do not expand link preview cards. The digest still uses `<https://url|link>` on every item line as a belt-and-braces rule.

**Manual runs:** Anyone can run `scripts/run_pipeline.bat` or `python scripts/run_pipeline.py` to collect and filter on demand. Use `scripts/refresh_digest_input.bat` to re-filter without re-collecting.

---

## 4. What the digest looks like

The digest uses a **title + thread** layout in Slack. The top-level channel message is the title only; all section content lives in a thread reply beneath it. **Empty sections are omitted** (if there are no qualifying papers that day, the Papers section does not appear).

**Top-level message (title only):**

```
📰 AI Engineering Digest - 2026-07-10
```

**Thread reply (all sections):**

```
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
- OpenClaw cron delivery mode is **`none`**: the agent posts via the **`message` tool** (title first, then thread); the agent's final reply is not delivered to Slack.
- The cron job must allow **`read`** and **`message`** tools (`--tools read,message`). Without `message`, delivery fails silently.

---

## 5. Model providers

| Provider | Model | Used for |
|----------|-------|----------|
| **Mistral** | Mistral Small (`mistral-small-latest`) | arXiv paper relevance scoring, editorial verify, pre-written `digest_description` in the filter step |
| **Mistral (OpenAI-compatible)** | `mistral-openai/mistral-small-latest` | OpenClaw cron digest generation (daily Slack post) |
| **OpenAI** | GPT-4o-mini (or upgraded ref via `openclaw doctor`) | Fallback option in the broader OpenClaw stack |

OpenClaw is configured to use Mistral via an OpenAI-compatible provider for daily digest generation.

---

## 6. Known limitations and pending work

| Area | Status |
|------|--------|
| **X (Twitter)** | **Active** via `twitter-cli` and `lib/twitter_collector.py`. Requires valid `TWITTER_AUTH_TOKEN` and `TWITTER_CT0` in `~/.config/morning-ai/.env`. AINews still adds indirect X coverage from smol.ai. |
| **Reddit** | Direct Reddit collection is geo-restricted from Nigeria. **AINews covers Reddit indirectly** via high-activity threads in smol.ai issues. |
| **Tavily** | Free tier credits can be exhausted. Collector code remains but **web search via Tavily is paused** until credits are renewed. |
| **Good AI List** | Requires **Playwright** and **Chromium** installed on the machine. If missing, Good AI List returns zero items but the rest of the pipeline still runs. |
| **Paths and OS** | Automation is set up for **Windows** (Task Scheduler, batch scripts). Cross-platform path handling exists in the pipeline; OpenClaw state lives under `~/.openclaw/`. |
| **Config location** | API keys and editorial profile live in `~/.config/morning-ai/` (legacy name from the upstream MorningAI project). OpenClaw config: `~/.openclaw/openclaw.json`. |

---

## 7. How to set up from scratch

For full setup (Python packages, API keys, Task Scheduler, OpenClaw cron, Slack channel), follow **[README.md](README.md)** in this folder.

Quick checklist for a new team member:

1. Install Python 3.9+ and optional packages (`mistralai`, `feedparser`, `playwright`, `tavily-python`, `twitter-cli` for X).
2. Copy `.env.example` to `~/.config/morning-ai/.env` and add keys (Mistral required, Twitter cookies for X, Tavily when renewed, GitHub token optional).
3. Create `~/.config/morning-ai/editorial_profile.md` with team editorial rules.
4. Configure OpenClaw **2026.7.1** with Slack (disable Telegram if unused), `unfurlLinks` / `unfurlMedia` false, and Mistral (see README).
5. Schedule OpenClaw gateway start at **7:30am** Lagos and `scripts/run_pipeline.bat` at **7:45am** Lagos time.
6. Confirm OpenClaw cron at **8:00am** reads `digest_input_today.json`, uses delivery mode **`none`**, and posts title + thread to Slack channel `C0BH9MD221H` via the `message` tool.

**Useful commands:**

```bash
# Full pipeline
python scripts/run_pipeline.py

# Re-filter only (after rule changes)
scripts\refresh_digest_input.bat

# Filter with stage breakdown
python scripts/filter_top_items.py --debug

# Sync digest prompt to OpenClaw after editing openclaw_digest_prompt.txt
python scripts/update_openclaw_cron.py

# Gateway status
openclaw gateway status
```

---

## Questions?

- **Pipeline logs:** `logs/pipeline.log`
- **What was already sent:** `logs/delivered_items.json`
- **Paper scores today:** `logs/paper_scores_today.json`
- **Rejected items today:** `logs/rejected_items_today.json`
- **Digest input for today:** `digest_input_today.json` (regenerated each run, not committed to git)
- **Developer details:** [README.md](README.md)
