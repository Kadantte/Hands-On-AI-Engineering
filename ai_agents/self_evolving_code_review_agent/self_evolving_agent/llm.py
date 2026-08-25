"""Ollama wrapper with schema-constrained outputs."""

from __future__ import annotations

import json
from typing import TypeVar

from ollama import Client
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LocalLLM:
    def __init__(self, host: str, model: str) -> None:
        self.host = host
        self.model = model
        self._client = Client(host=host)

    def check(self) -> tuple[bool, str]:
        try:
            response = self._client.list()
        except Exception as exc:
            return False, f"Cannot reach Ollama at {self.host}: {exc}"

        names = {item.model for item in response.models if item.model}
        if self.model in names:
            return True, f"Ollama ready with {self.model}"
        return False, f"Model '{self.model}' is missing. Run: ollama pull {self.model}"

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 1800,
    ) -> T:
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema.model_json_schema(),
            think=False,
            options={"temperature": 0.1, "num_predict": max_tokens},
        )
        content = (response.message.content or "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty structured response.")
        return schema.model_validate(json.loads(content))

