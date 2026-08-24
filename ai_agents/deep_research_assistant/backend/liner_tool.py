import os

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

LINER_SEARCH_URL = "https://platform.liner.com/api/v1/tools/search/web"
DEFAULT_MAX_RESULTS = 10


def _format_result(index: int, item: dict) -> str:
    title = item.get("title") or "Untitled"
    url = item.get("url") or "No URL provided"
    description = item.get("description") or "No description provided"
    date = item.get("date") or "No date provided"
    return (
        f"[{index}] Title: {title}\n"
        f"    URL: {url}\n"
        f"    Description: {description}\n"
        f"    Date: {date}"
    )


@tool
def liner_search(query: str) -> str:
    """Search the web using the Liner Web Search API.

    Call this tool to gather current, citable sources for a research question.
    Returns numbered results with title, url, description, and date so each
    claim in the final report can be cited by result number.

    Args:
        query: The search query to look up.
    """
    api_key = os.getenv("LINER_API_KEY")
    if not api_key:
        return (
            "Error: LINER_API_KEY is not set. Add it to the backend .env file "
            "and try the search again."
        )

    try:
        response = requests.post(
            LINER_SEARCH_URL,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={"query": query, "max_results": DEFAULT_MAX_RESULTS},
            timeout=30,
        )
    except requests.RequestException as exc:
        return f"Error: failed to reach the Liner Web Search API ({exc})."

    if not response.ok:
        return (
            f"Error: Liner Web Search API returned HTTP {response.status_code}. "
            f"Response: {response.text}"
        )

    try:
        payload = response.json()
    except ValueError:
        return "Error: Liner Web Search API returned a non-JSON response."

    results = payload.get("results") or []
    if not results:
        return f"No search results found for query: {query}"

    formatted = [
        _format_result(index, item) for index, item in enumerate(results, start=1)
    ]
    header = f"Liner web search results for: {query}\nCite sources by number, title, and URL.\n"
    return header + "\n\n".join(formatted)
