from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_mistralai import ChatMistralAI

from liner_tool import liner_search

load_dotenv()

SYSTEM_PROMPT = (
    "You are a deep research assistant. For every question you receive, "
    "plan your research steps using write_todos, search for information "
    "using the liner_search tool, synthesize the results into a "
    "comprehensive answer where every claim is backed by a source from "
    "the search results, and save the final cited report as a markdown "
    "file to the filesystem. Never make claims without citing a source "
    "from the search results."
)

model = ChatMistralAI(model="mistral-small-latest")

agent = create_deep_agent(
    model=model,
    tools=[liner_search],
    middleware=[TodoListMiddleware()],
    system_prompt=SYSTEM_PROMPT,
)
