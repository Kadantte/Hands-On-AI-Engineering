import os
from dotenv import load_dotenv
from crewai import Crew, Process, LLM

from agents import (
    trend_intelligence_agent,
    copywriter_agent,
    visual_strategist_agent,
    posting_strategist_agent
)
from tasks import (
    trend_research_task,
    caption_writing_task,
    visual_prompt_task,
    posting_strategy_task
)

load_dotenv()

llm = LLM(
    model="moonshotai/Kimi-K3",
    base_url="https://inference.baseten.co/v1",
    api_key=os.getenv("BASETEN_API_KEY"),
    max_tokens=500,
    custom_openai=True,
    extra_body={"reasoning_effort": "none"},
)


def build_instagram_post_crew():
    return Crew(
        agents=[
            trend_intelligence_agent,
            copywriter_agent,
            visual_strategist_agent,
            posting_strategist_agent
        ],
        tasks=[
            trend_research_task,
            caption_writing_task,
            visual_prompt_task,
            posting_strategy_task
        ],
        process=Process.sequential,
        verbose=True
    )


def run_instagram_post_crew(topic: str):
    crew = build_instagram_post_crew()
    result = crew.kickoff(inputs={"topic": topic})
    return result, crew
