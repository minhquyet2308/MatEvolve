"""Common base for the prompt-based agents."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..llm import LLMClient, Message, extract_json

log = logging.getLogger(__name__)


class Agent:
    """An agent is a language model plus a fixed system prompt.

    Subclasses build the user message(s) for their task and parse the reply.
    ``temperature`` defaults to the client's (0.5 in the manuscript) so the
    same decoding setting applies to every agent unless overridden.
    """

    def __init__(self, llm: LLMClient, system_prompt: str, temperature: Optional[float] = None) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.temperature = temperature

    def messages(self, user: str) -> List[Message]:
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

    def call(self, messages: List[Message]) -> str:
        return self.llm.complete(messages, temperature=self.temperature)

    def call_json(self, messages: List[Message], retries: int = 1):
        """Call the model and parse JSON; on failure ask once more for valid JSON."""
        reply = self.call(messages)
        for attempt in range(retries + 1):
            try:
                return extract_json(reply), reply
            except ValueError:
                if attempt == retries:
                    raise
                log.info("%s: reply was not valid JSON; asking again", type(self).__name__)
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "Return ONLY valid JSON, no prose, no code fences."},
                ]
                reply = self.call(messages)
        raise RuntimeError("unreachable")
