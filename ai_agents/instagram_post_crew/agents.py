import os
from dotenv import load_dotenv
from crewai import Agent, LLM

from tools import duckduckgo_search_tool, firecrawl_scrape_tool

load_dotenv()

llm = LLM(
    model="moonshotai/Kimi-K3",
    base_url="https://inference.baseten.co/v1",
    api_key=os.getenv("BASETEN_API_KEY"),
    max_tokens=500,
    custom_openai=True,
    extra_body={"reasoning_effort": "none"},
)

trend_intelligence_agent = Agent(
    role="Trend Intelligence Agent",
    goal=(
        "Do at most 2 searches, then immediately produce a short structured creative brief "
        "for Instagram on the given topic. Prefer finishing over more research."
    ),
    backstory=(
        "You are a social media trend analyst. You gather just enough signal to explain "
        "why formats, hooks, and visuals are working, then you stop searching and write "
        "a concise brief. You never run long research loops."
    ),
    tools=[duckduckgo_search_tool, firecrawl_scrape_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    max_retry_limit=1
)

copywriter_agent = Agent(
    role="Copywriter Agent",
    goal=(
        "Write three short caption variants (educational, emotional, provocative), pick "
        "the best one for the niche, and include hashtags. Keep the whole answer concise."
    ),
    backstory=(
        "You are a direct response copywriter for short form Instagram captions. "
        "You write tight copy and choose one clear winner quickly."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=1,
    max_retry_limit=1
)

visual_strategist_agent = Agent(
    role="Visual Strategist Agent",
    goal=(
        "Write one short reasoning note plus one ready to use image generation prompt "
        "grounded in trending aesthetics for the topic."
    ),
    backstory=(
        "You are a visual creative director. You briefly justify color and style choices, "
        "then deliver one detailed image prompt."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=1,
    max_retry_limit=1
)

posting_strategist_agent = Agent(
    role="Posting Strategist Agent",
    goal=(
        "Recommend one best day and time to post for this specific niche, with brief "
        "audience-behavior reasoning."
    ),
    backstory=(
        "You are a social media scheduling strategist. You give niche-specific timing "
        "advice in a short, decisive answer."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=1,
    max_retry_limit=1
)
