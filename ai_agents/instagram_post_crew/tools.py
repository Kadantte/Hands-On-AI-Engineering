import os
from dotenv import load_dotenv
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import FirecrawlScrapeWebsiteTool

load_dotenv()

_duckduckgo_search_run = DuckDuckGoSearchRun()


@tool("DuckDuckGo Search")
def duckduckgo_search_tool(query: str) -> str:
    """Search DuckDuckGo for the given query and return the top results as text."""
    return _duckduckgo_search_run.run(query)


firecrawl_scrape_tool = FirecrawlScrapeWebsiteTool(
    api_key=os.getenv("FIRECRAWL_API_KEY")
)
