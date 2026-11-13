"""Voice-intent parser with deterministic offline NLP and optional LLM adapter."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from vector_lab.types import Blackboard, IntentGoal, IntentType, Pose2D

_COORD_PATTERN = re.compile(
    r"(?:x\s*=\s*(-?\d+(?:\.\d+)?)\s*[,;\s]+y\s*=\s*(-?\d+(?:\.\d+)?))"
    r"|(?:\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\))"
    r"|(?:to\s+(-?\d+(?:\.\d+)?)\s*[,;\s]+(-?\d+(?:\.\d+)?))",
    re.IGNORECASE,
)

_GREET_NAME_PATTERN = re.compile(
    r"(?:greet|say\s+(?:hi|hello)(?:\s+to)?|hello|hi)\s+([a-zA-Z][a-zA-Z0-9_-]{1,24})",
    re.IGNORECASE,
)


class IntentAdapter(ABC):
    """Abstract voice/text utterance to structured IntentGoal translator."""

    @abstractmethod
    def parse(self, utterance: str) -> IntentGoal:
        """Convert raw natural-language command into a structured IntentGoal."""


class DeterministicIntentParser(IntentAdapter):
    """Offline rule-based parser mapping natural-language commands to Behavior Tree goals."""

    def parse(self, utterance: str) -> IntentGoal:
        cleaned = utterance.strip()
        lower = cleaned.lower()

        if any(kw in lower for kw in ("stop", "halt", "freeze", "cancel", "abort")):
            return IntentGoal(
                intent=IntentType.STOP,
                utterance=cleaned,
                confidence=0.99,
                source="deterministic",
            )

        if any(kw in lower for kw in ("dock", "charger", "charge", "go home", "recharge")):
            return IntentGoal(
                intent=IntentType.RETURN_TO_DOCK,
                utterance=cleaned,
                confidence=0.97,
                source="deterministic",
            )

        coord_match = _COORD_PATTERN.search(cleaned)
        if coord_match is not None:
            groups = [g for g in coord_match.groups() if g is not None]
            if len(groups) >= 2:
                tx, ty = float(groups[0]), float(groups[1])
                return IntentGoal(
                    intent=IntentType.GOTO_POSE,
                    utterance=cleaned,
                    target_pose=Pose2D(x=tx, y=ty, theta=0.0),
                    confidence=0.96,
                    source="deterministic",
                )

        if any(kw in lower for kw in ("greet", "say hi", "say hello", "hello", "hi ")):
            name_match = _GREET_NAME_PATTERN.search(cleaned)
            person = name_match.group(1) if name_match else "friend"
            if person.lower() in {"vector", "there", "please", "now"}:
                person = "friend"
            return IntentGoal(
                intent=IntentType.GREET_PERSON,
                utterance=cleaned,
                person_name=person,
                confidence=0.92,
                source="deterministic",
            )

        if any(kw in lower for kw in ("status", "battery", "report", "health")):
            return IntentGoal(
                intent=IntentType.STATUS_REPORT,
                utterance=cleaned,
                confidence=0.94,
                source="deterministic",
            )

        if any(kw in lower for kw in ("patrol", "explore", "resume", "map", "scan", "wander")):
            return IntentGoal(
                intent=IntentType.PATROL,
                utterance=cleaned,
                confidence=0.95,
                source="deterministic",
            )

        return IntentGoal(
            intent=IntentType.STATUS_REPORT,
            utterance=cleaned,
            confidence=0.45,
            source="deterministic-fallback",
        )


class LLMIntentAdapter(IntentAdapter):
    """Optional OpenAI-compatible HTTP LLM intent parser with deterministic offline fallback."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout_s: float = 3.0,
        fallback: IntentAdapter | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.fallback = fallback or DeterministicIntentParser()

    def parse(self, utterance: str) -> IntentGoal:
        if not self.endpoint_url:
            return self.fallback.parse(utterance)

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify robot command into JSON with keys: "
                        "intent (PATROL|GOTO_POSE|RETURN_TO_DOCK|GREET_PERSON|STOP|STATUS_REPORT), "
                        "x (float|null), y (float|null), person_name (string|null)."
                    ),
                },
                {"role": "user", "content": utterance},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = Request(
                self.endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            intent_type = IntentType(str(parsed["intent"]).upper())
            target_pose = None
            if parsed.get("x") is not None and parsed.get("y") is not None:
                target_pose = Pose2D(float(parsed["x"]), float(parsed["y"]), 0.0)
            return IntentGoal(
                intent=intent_type,
                utterance=utterance.strip(),
                target_pose=target_pose,
                person_name=parsed.get("person_name"),
                confidence=0.95,
                source="llm-adapter",
            )
        except (URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
            return self.fallback.parse(utterance)


class VoiceIntentPipeline:
    """Dispatches parsed natural-language commands onto the Behavior Tree blackboard."""

    def __init__(self, adapter: IntentAdapter | None = None) -> None:
        self.adapter = adapter or DeterministicIntentParser()

    def dispatch(self, utterance: str, blackboard: Blackboard) -> IntentGoal:
        """Parse utterance and post goal onto blackboard.pending_intent."""
        goal = self.adapter.parse(utterance)
        if goal.intent != IntentType.STOP:
            blackboard.halted_by_operator = False
        blackboard.pending_intent = goal
        blackboard.log_event(f"INTENT:{goal.intent.value} ({goal.utterance})")
        return goal
