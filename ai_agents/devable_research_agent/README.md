# Devable Research Agent

> Automated daily AI engineering digest that collects signals from arXiv, GitHub, HuggingFace, Good AI List, AINews, and HN then posts a curated digest to Slack via OpenClaw.

## Overview

Devable Research Agent solves the problem of staying current on AI engineering news without manually checking dozens of sources every morning. It runs a unattended 3-step pipeline on a schedule: collect raw signals, filter and rank the best items, then deliver a structured Slack digest.

The pipeline is built for builders who care about papers, repos, model drops, tools, and community signals. Output is tuned for Slack with anti-unfurl link formatting and editorial rules stored outside the repo.

**Who benefits:** AI engineers, ML platform teams, and technical leads who want a daily, curated briefing in Slack without maintaining their own scrapers.

## Features

- **Multi-source collection:** arXiv, GitHub, HuggingFace, Hacker News, Tavily web search, Good AI List, and AINews (smol.ai)
- **Smart filtering:** Category quotas, Mistral paper scoring, frequency caps, and editorial dedup against recent deliveries
- **Slack-ready digest:** OpenClaw cron reads filtered JSON plus editorial profile and posts to your channel
- **Entity-centric tracking:** 80+ curated entities across markdown registries (labs, infra, coding agents, KOLs)
- **Windows automation:** Task Scheduler entry point via batch script; OpenClaw handles delivery at a fixed time

## Tech Stack

**Core:**
- Python 3.9+ (stdlib-first collectors; optional packages for Tavily, AINews RSS, Good AI List, Mistral)
- OpenClaw (cron agent for Slack digest generation and delivery)

**Collectors & APIs:**
- arXiv API, GitHub API, HuggingFace API, Hacker News (Algolia)
- Tavily Search API (`tavily-python`)
- AINews RSS + HTML parsing (`feedparser`)
- Good AI List scraper (`playwright` + Chromium)

**Model providers:**
- **Mistral** (`mistral-small-latest`): paper relevance scoring, editorial verify step, and OpenClaw digest generation
- **Gemini** (planned): native X/Twitter collector when restored

## Prerequisites

Before you begin, ensure you have:

- Python 3.9 or higher
- [OpenClaw](https://github.com/openclaw/openclaw) installed and configured with Slack
- API keys for:
  - [ ] **Mistral** (required for filter scoring and digest)
  - [ ] **Tavily** (required for web search collector)
  - [ ] **GitHub** (optional, higher rate limits for GitHub collector)
- Optional Python packages:
  - `pip install mistralai tavily-python feedparser playwright`
  - `playwright install chromium` (for Good AI List)
- Config directory: `~/.config/morning-ai/` (env file and editorial profile)
- Basic understanding of scheduled tasks (Windows Task Scheduler) or cron on your OS

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/devable_research_agent
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

There is no single `requirements.txt`. Install optional packages used by collectors and the filter step:

```bash
pip install mistralai tavily-python feedparser playwright
playwright install chromium
```

### 4. Set Up Environment Variables

Copy the example env file to the global config location (recommended):

```bash
# Windows
mkdir %USERPROFILE%\.config\morning-ai
copy .env.example %USERPROFILE%\.config\morning-ai\.env

# macOS/Linux
mkdir -p ~/.config/morning-ai
cp .env.example ~/.config/morning-ai/.env
```

Edit `~/.config/morning-ai/.env` and add your keys:

```bash
TAVILY_API_KEY=tvly-your-key-here
MISTRAL_API_KEY=your-mistral-key-here
GITHUB_TOKEN=ghp_optional_for_higher_limits
```

Create your editorial rules file:

```bash
# ~/.config/morning-ai/editorial_profile.md
# Describe tone, topics to prioritize, and items to skip.
```

OpenClaw also needs `MISTRAL_API_KEY` in `~/.openclaw/.env` for digest generation.

## Usage

### How It Works (3-Step Pipeline)

```
Step 1 - Collect (7:45am)
  scripts/run_pipeline.py
    -> skills/tracking-list/scripts/collect.py
    -> data_today.json

Step 2 - Filter
  scripts/filter_top_items.py
    -> digest_input_today.json
    -> logs/delivered_items.json (frequency cap log)

Step 3 - Deliver (8:00am)
  OpenClaw cron reads digest_input_today.json
    + ~/.config/morning-ai/editorial_profile.md
    + scripts/openclaw_digest_prompt.txt
    -> Slack channel
```

### Sources Covered

| Source | Module | Notes |
|--------|--------|-------|
| arXiv | `lib/arxiv.py` | Entity-driven queries |
| GitHub | `lib/github.py` | Releases and activity; token optional |
| HuggingFace | `lib/huggingface.py` | Models and datasets |
| Hacker News | `lib/hackernews.py` | Algolia search |
| Tavily | `lib/tavily_collector.py` | Web signals; requires API key |
| Good AI List | `lib/goodailist_scraper.py` | Trending repos; Playwright |
| AINews | `lib/ainews_collector.py` | smol.ai RSS; X and Reddit sections parsed |

### Run Manually

From the project root:

```bash
# Full pipeline (collect + filter)
python scripts/run_pipeline.py

# Collect only
python skills/tracking-list/scripts/collect.py --date 2026-07-10 --depth default --no-cache --output data_today.json

# Re-filter existing data (after rule changes)
python scripts/filter_top_items.py data_today.json digest_input_today.json --top 20

# Or on Windows
scripts\run_pipeline.bat
scripts\refresh_digest_input.bat
```

Sync the OpenClaw digest prompt after editing `scripts/openclaw_digest_prompt.txt`:

```bash
python scripts/update_openclaw_cron.py
```

### Example Output

**`digest_input_today.json` (excerpt):** filtered items with title, summary, source, source_url, importance, and category labels for OpenClaw routing.

**Slack digest sections:**
- Top Papers
- Trending Repos
- What's Trending on X (AINews items with x.com URLs)
- What Happened on Reddit (AINews Reddit threads)
- New Model Drops
- New Tools

## Automation

### Windows Task Scheduler (collection + filter)

1. Schedule `scripts/run_pipeline.bat` daily at **7:45am** (adjust timezone as needed).
2. Update `PROJECT_ROOT` inside `run_pipeline.bat` to your checkout path.
3. Pipeline writes `logs/pipeline.log` for run history.

### OpenClaw Cron (Slack delivery)

1. Configure an OpenClaw cron job to run at **8:00am** (after collection finishes).
2. Job reads `digest_input_today.json` from the project root.
3. Prompt source of truth: `scripts/openclaw_digest_prompt.txt` (sync via `update_openclaw_cron.py`).
4. Model: `mistral-openai/mistral-small-latest` (OpenAI-compatible provider with `supportsStore: false` for Mistral API).

## Known Limitations

- **Reddit collector:** Not included in the minimal pipeline. AINews still surfaces high-activity Reddit threads from smol.ai issues.
- **Native X/Twitter collector:** Removed from the active collector set. X signals currently come from AINews Twitter sections. A dedicated X collector using Gemini is planned for a future release.
- **Good AI List:** Requires Playwright and Chromium. Collection fails gracefully if Playwright is not installed.
- **Hardcoded paths:** `run_pipeline.bat` may need `PROJECT_ROOT` updated for your machine. OpenClaw cron job ID in `update_openclaw_cron.py` is environment-specific.
- **Config path legacy:** Runtime config still uses `~/.config/morning-ai/` from the upstream MorningAI fork.

## Project Structure

```
devable_research_agent/
├── scripts/
│   ├── run_pipeline.py           # Orchestrator (collect + filter)
│   ├── run_pipeline.bat          # Task Scheduler entry
│   ├── refresh_digest_input.bat  # Re-filter only
│   ├── filter_top_items.py       # Scoring, quotas, digest input
│   ├── openclaw_digest_prompt.txt
│   └── update_openclaw_cron.py
├── skills/tracking-list/scripts/
│   └── collect.py                # Step 1 collector orchestrator
├── lib/                          # Collectors, scoring, dedupe, entities
├── entities/                     # Tracked entity registries (*.md)
├── logs/
│   └── .gitkeep
├── .env.example
├── .gitignore
└── README.md
```

## How It Works

**Collection:** `collect.py` runs seven collectors concurrently, then classifies, scores, deduplicates, and cross-links items into `data_today.json`.

**Filtering:** `filter_top_items.py` applies date cutoffs, maintenance filters, Mistral paper scoring, category quotas (papers, repos, web, X via AINews, HN), and a Mistral editorial verify pass against `logs/delivered_items.json`. Output is a smaller JSON file for OpenClaw.

**Delivery:** OpenClaw applies `editorial_profile.md` and section routing rules from the prompt file, then posts the formatted digest to Slack with `<url|link>` syntax to avoid unfurl cards.

[Back to Top](#devable-research-agent)
