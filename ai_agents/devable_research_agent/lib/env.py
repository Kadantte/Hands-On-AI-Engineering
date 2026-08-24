"""Environment and configuration management for devable-research-agent."""

import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_env_file(path: str) -> Dict[str, str]:
    """Parse a .env file into a dict."""
    env = {}
    p = Path(path)
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        env[key] = value
    return env


def get_config() -> Dict[str, Any]:
    """Load config from environment + .env files.

    Priority: env vars > project .env > project .claude/morning-ai.env > global config
    """
    global_env = load_env_file(str(Path.home() / ".config" / "morning-ai" / ".env"))
    project_claude_env = load_env_file(".claude/morning-ai.env")
    project_env = load_env_file(".env")
    local_env = load_env_file(".env.local")

    config = {}
    config.update(global_env)
    config.update(project_claude_env)
    config.update(project_env)
    config.update(local_env)
    config.update({k: v for k, v in os.environ.items() if k.startswith((
        "MORNING_AI_", "GITHUB_", "TAVILY_", "OPENAI_", "MISTRAL_", "ANTHROPIC_", "TWITTER_",
    ))})

    return config


def get_key(config: Dict[str, Any], key: str) -> Optional[str]:
    """Get a config value, returning None if empty."""
    val = config.get(key, "")
    return val if val else None


def get_available_sources(config: Dict[str, Any]) -> Dict[str, bool]:
    """Check which data sources are available based on configured API keys."""
    return {
        "hackernews": True,
        "github": bool(get_key(config, "GITHUB_TOKEN")),
        "huggingface": True,
        "arxiv": True,
        "tavily": bool(get_key(config, "TAVILY_API_KEY")),
        "goodailist": True,
        "ainews": True,
        "x": bool(get_key(config, "TWITTER_AUTH_TOKEN") and get_key(config, "TWITTER_CT0")),
    }
