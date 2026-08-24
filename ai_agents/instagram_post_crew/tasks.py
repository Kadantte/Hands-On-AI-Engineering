from dotenv import load_dotenv
from crewai import Task

from agents import (
    trend_intelligence_agent,
    copywriter_agent,
    visual_strategist_agent,
    posting_strategist_agent
)

load_dotenv()

trend_research_task = Task(
    description=(
        "Research Instagram trends for the topic: '{topic}'. "
        "Use at most 2 tool calls total, then stop researching. "
        "Write a SHORT creative brief with these sections: "
        "1. Trending Formats, 2. Winning Hooks, 3. Visual Style Patterns, 4. Why These Work. "
        "Keep the entire brief under 350 words."
    ),
    expected_output=(
        "A short markdown creative brief with four labelled sections: "
        "Trending Formats, Winning Hooks, Visual Style Patterns, and Why These Work. "
        "Under 350 words total."
    ),
    agent=trend_intelligence_agent
)

caption_writing_task = Task(
    description=(
        "Using the creative brief for '{topic}', write three short Instagram caption "
        "variants: educational, emotional, and provocative. Briefly say which fits the "
        "niche best, then output the selected final caption with hashtags. "
        "Keep the entire response under 250 words."
    ),
    expected_output=(
        "Three labelled caption variants, one short selection reason, and one final "
        "caption with hashtags. Under 250 words total."
    ),
    agent=copywriter_agent,
    context=[trend_research_task]
)

visual_prompt_task = Task(
    description=(
        "Using the creative brief for '{topic}', briefly note color psychology and "
        "visual style choices, then give one ready to use image generation prompt. "
        "Keep the entire response under 200 words."
    ),
    expected_output=(
        "A short reasoning note followed by one detailed image generation prompt. "
        "Under 200 words total."
    ),
    agent=visual_strategist_agent,
    context=[trend_research_task]
)

posting_strategy_task = Task(
    description=(
        "Using the creative brief for '{topic}', recommend the best day and time to "
        "publish. Tie the reason to this niche's audience behavior. "
        "Keep the entire response under 150 words."
    ),
    expected_output=(
        "A short niche-specific reasoning note and one clear day/time recommendation. "
        "Under 150 words total."
    ),
    agent=posting_strategist_agent,
    context=[trend_research_task]
)
