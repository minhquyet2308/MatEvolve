"""Language-model clients.

All MatEvolve agents talk to a language model through the small
:class:`LLMClient` interface below: a list of chat messages in, a string
out.  The reported experiments used a single frozen commercial model
(Claude Fable 5, API identifier ``claude-fable-5``) at decoding temperature
0.5 for every agent; :class:`AnthropicClient` reproduces that setting.
:class:`OpenAICompatibleClient` targets any server that speaks the
OpenAI chat-completions protocol, so the framework can be re-run with other
hosted or self-hosted models.  :class:`RecordingClient` and
:class:`ScriptedClient` support logging and offline tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

Message = Dict[str, str]
log = logging.getLogger(__name__)

#: Model and decoding temperature used for every agent in the reported runs.
DEFAULT_MODEL = "claude-fable-5"
DEFAULT_TEMPERATURE = 0.5


class LLMClient(ABC):
    """Minimal chat interface shared by every agent."""

    def __init__(self, model: str, temperature: float = DEFAULT_TEMPERATURE) -> None:
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def complete(self, messages: Sequence[Message], temperature: Optional[float] = None) -> str:
        """Return the assistant reply for an OpenAI-style message list.

        ``messages`` holds dicts with ``role`` in {system, user, assistant}
        and ``content``.  Implementations must raise on hard failures so the
        calling loop can decide whether to retry.
        """

    # convenience -----------------------------------------------------------
    def chat(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        return self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )


def _split_system(messages: Sequence[Message]):
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    return "\n\n".join(system_parts), rest


class EmptyReply(RuntimeError):
    """The model returned no text; treated as a transient error and retried."""


class AnthropicClient(LLMClient):
    """Claude models through the official ``anthropic`` SDK.

    Parameters
    ----------
    model:
        API identifier; the reported runs used ``claude-fable-5``.
    temperature:
        Decoding temperature; 0.5 in the reported runs.  SDK versions >= 1.0
        no longer expose it on ``Messages.create``; it is then passed through
        ``extra_body`` and, if the API of the model rejects it, dropped with a
        warning (the model default applies).
    max_tokens:
        Generation budget.  CIF files for 100-atom cells need several
        thousand tokens, hence the generous default.
    api_key:
        Defaults to the ``ANTHROPIC_API_KEY`` environment variable.
    max_retries:
        Retries on transient API errors with exponential back-off.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 16000,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(model, temperature)
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install anthropic") from exc
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout
        )
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    _temperature_unsupported = False       # set once the API has rejected the parameter (class-wide)

    def _supports_temperature_kwarg(self) -> bool:
        """``anthropic`` < 1.0 exposes ``temperature`` on ``Messages.create``; later SDKs do not."""
        import inspect
        try:
            return "temperature" in inspect.signature(self._client.messages.create).parameters
        except (TypeError, ValueError):  # pragma: no cover
            return False

    def complete(self, messages: Sequence[Message], temperature: Optional[float] = None) -> str:
        system, rest = _split_system(messages)
        t = self.temperature if temperature is None else temperature
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": m["role"], "content": m["content"]} for m in rest],
        )
        if system:
            kwargs["system"] = system
        if t is not None and not self._temperature_unsupported:
            if self._supports_temperature_kwarg():
                kwargs["temperature"] = t
            else:
                kwargs["extra_body"] = {"temperature": t}      # SDK >= 1.0: pass it through to the API
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(**kwargs)
                text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip()
                if text:
                    return text
                # no text block (e.g. the model stopped before writing any): a transient failure, retried below
                raise EmptyReply(f"empty reply (stop_reason={getattr(resp, 'stop_reason', None)!r}, "
                                 f"{len(resp.content)} content blocks)")
            except TypeError:
                raise                                          # a programming error, not a transient API failure
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status == 400 and "temperature" in str(exc).lower() and ("temperature" in kwargs or "extra_body" in kwargs):
                    # the Messages API of this model does not accept a sampling temperature
                    AnthropicClient._temperature_unsupported = True
                    kwargs.pop("temperature", None)
                    kwargs.pop("extra_body", None)
                    log.warning("the Messages API of %s does not accept a 'temperature' parameter; the same model is "
                                "used with its built-in default sampling (requested T=%.2f cannot be set)", self.model, t)
                    continue
                if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                    raise                                      # bad request / auth: retrying cannot help
                if attempt == self.max_retries - 1:
                    raise
                log.warning("Anthropic API error (%s); retry %d in %.0fs", exc, attempt + 1, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")


class OpenAICompatibleClient(LLMClient):
    """Any endpoint implementing the OpenAI chat-completions protocol.

    Covers the OpenAI API itself as well as self-hosted servers (vLLM,
    llama.cpp, LM Studio, ...) that expose ``/v1/chat/completions``.
    """

    def __init__(
        self,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 16000,
        max_retries: int = 5,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(model, temperature)
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install openai") from exc
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
            timeout=timeout,
        )
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    _temperature_unsupported = False       # set once the API has rejected the parameter (class-wide)

    def complete(self, messages: Sequence[Message], temperature: Optional[float] = None) -> str:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                log.warning("OpenAI-compatible API error (%s); retry %d in %.0fs", exc, attempt + 1, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")


class RecordingClient(LLMClient):
    """Wraps another client and appends every exchange to a JSONL log.

    Every agent conversation of a run is therefore auditable after the fact
    (the ``conversation.jsonl`` files referred to in the documentation).
    """

    def __init__(self, inner: LLMClient, log_path: Path, step: str = "") -> None:
        super().__init__(inner.model, inner.temperature)
        self.inner = inner
        self.log_path = Path(log_path)
        self.step = step
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def complete(self, messages: Sequence[Message], temperature: Optional[float] = None) -> str:
        reply = self.inner.complete(messages, temperature=temperature)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.log_path.open("a", encoding="utf-8") as fh:
            for m in messages:
                fh.write(json.dumps({"step": self.step, "role": m["role"], "content": m["content"], "ts": ts}) + "\n")
            fh.write(json.dumps({"step": self.step, "role": "assistant", "content": reply, "ts": ts}) + "\n")
        return reply


class NeedReply(Exception):
    """Raised by :class:`AgentClient` when a model reply has to be supplied by the agent session.

    The caller (a translation chain, a design step) stops; the orchestrating
    agent answers the request file and re-runs the command, which replays all
    earlier replies and continues from here.
    """

    def __init__(self, request_path: Path, request_id: str, step: str) -> None:
        super().__init__(f"model reply needed: {request_path}")
        self.request_path = Path(request_path)
        self.request_id = request_id
        self.step = step


class AgentClient(LLMClient):
    """The skill transport: model replies are written by the Claude session that runs the skill.

    Same code path as the API clients -- every agent calls ``complete(messages)`` --
    but the reply is exchanged through files under ``exchange``::

        pending/<id>.json   a request the agent has to answer (system prompt, messages, step)
        replies/<id>.txt    the reply, written by the agent (the orchestrator or a worker sub-agent)
        conversations.jsonl every answered exchange, same format as RecordingClient

    ``<id>`` is a hash of the message sequence, so re-running a command replays the
    replies already given (deterministically, the code asks the same questions in
    the same order) and raises :class:`NeedReply` at the first unanswered one.  The
    same request asked twice (identical messages) receives the same reply.
    """

    def __init__(self, exchange: Path, model: str = "agent", temperature: float = DEFAULT_TEMPERATURE) -> None:
        super().__init__(model, temperature)
        self.exchange = Path(exchange)
        self.pending_dir = self.exchange / "pending"
        self.replies_dir = self.exchange / "replies"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.replies_dir.mkdir(parents=True, exist_ok=True)
        self.step = ""                      # label of the current step (set by the caller for the request files)
        self.chain = ""                     # label of the current chain (structure id / design step)
        self.requested: List[Path] = []     # requests raised during this process

    @staticmethod
    def request_id(messages: Sequence[Message]) -> str:
        payload = json.dumps([{"role": m["role"], "content": m["content"]} for m in messages], ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def complete(self, messages: Sequence[Message], temperature: Optional[float] = None) -> str:
        rid = self.request_id(messages)
        reply_path = self.replies_dir / f"{rid}.txt"
        if reply_path.is_file():
            reply = reply_path.read_text(encoding="utf-8").strip()
            if reply:
                self._log_once(rid, messages, reply)
                pend = self.pending_dir / f"{rid}.json"
                if pend.is_file():
                    pend.unlink()
                return reply
        req = self.pending_dir / f"{rid}.json"
        if not req.is_file():
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            req.write_text(json.dumps({
                "id": rid, "step": self.step, "chain": self.chain,
                "reply_file": str(reply_path),
                "instructions": "Act as the language model of this agent: read the system prompt and the conversation, "
                                "and write the next assistant reply -- the reply text only, no commentary -- to reply_file.",
                "system": system,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"],
            }, indent=1, ensure_ascii=False), encoding="utf-8")
        self.requested.append(req)
        raise NeedReply(req, rid, self.step)

    def _log_once(self, rid: str, messages: Sequence[Message], reply: str) -> None:
        used = self.exchange / "used.txt"
        seen = set(used.read_text().split()) if used.is_file() else set()
        if rid in seen:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with (self.exchange / "conversations.jsonl").open("a", encoding="utf-8") as fh:
            for m in messages:
                fh.write(json.dumps({"step": self.step, "chain": self.chain, "role": m["role"], "content": m["content"], "ts": ts}) + "\n")
            fh.write(json.dumps({"step": self.step, "chain": self.chain, "role": "assistant", "content": reply, "ts": ts}) + "\n")
        with used.open("a") as fh:
            fh.write(rid + "\n")

    def pending(self) -> List[Path]:
        """Every request still waiting for a reply (across runs)."""
        return sorted(self.pending_dir.glob("*.json"))


class ScriptedClient(LLMClient):
    """Offline client for tests: replies come from a queue or a callback."""

    def __init__(self, replies: Optional[Sequence[str]] = None, responder: Optional[Callable[[Sequence[Message]], str]] = None) -> None:
        super().__init__("scripted", 0.0)
        self._replies = list(replies or [])
        self._responder = responder
        self.calls: List[List[Message]] = []

    def complete(self, messages: Sequence[Message], temperature: Optional[float] = None) -> str:
        self.calls.append([dict(m) for m in messages])
        if self._responder is not None:
            return self._responder(messages)
        if not self._replies:
            raise RuntimeError("ScriptedClient ran out of replies")
        return self._replies.pop(0)


@dataclass
class LLMConfig:
    """Serializable description of a client (see ``configs/*.yaml``)."""

    provider: str = "anthropic"          # anthropic | openai | agent (replies supplied by the Claude session running a skill)
                                         # | dummy (request-derived placeholder replies, no model: smoke tests)
    exchange: str = "runs/matevolve/llm"  # provider agent: folder of the request / reply files
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = 16000
    base_url: Optional[str] = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    extra: Dict[str, object] = field(default_factory=dict)

    def build(self) -> LLMClient:
        api_key = os.environ.get(self.api_key_env)
        if self.provider == "anthropic":
            return AnthropicClient(self.model, self.temperature, self.max_tokens, api_key=api_key)
        if self.provider == "openai":
            return OpenAICompatibleClient(self.model, self.temperature, base_url=self.base_url, api_key=api_key, max_tokens=self.max_tokens)
        if self.provider == "agent":
            return AgentClient(Path(self.exchange), self.model, self.temperature)
        if self.provider == "dummy":
            from .dummy_llm import DummyClient

            return DummyClient()
        raise ValueError(f"unknown LLM provider {self.provider!r}")


def strip_code_fences(text: str) -> str:
    """Remove a surrounding ```...``` fence if the model added one."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        t = t[first_nl + 1 :] if first_nl != -1 else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str, expect: type = dict):
    """Parse the first JSON object (or array) found in ``text``.

    Models occasionally wrap JSON in prose or fences, or copy the trailing
    comma of the design-prompt example (``],\n}``); this is the single
    tolerant parser every agent uses.  Raises ``ValueError`` on failure or
    when the parsed value is not of type ``expect``.
    """
    t = strip_code_fences(text)
    candidates = [t]
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = t.find(opener), t.rfind(closer)
        if start != -1 and end > start:
            candidates.append(t[start : end + 1])
    for cand in candidates:
        for attempt in (cand, _TRAILING_COMMA.sub(r"\1", cand)):
            try:
                obj = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, expect):
                return obj
    raise ValueError(f"no JSON {expect.__name__} found in model reply")
