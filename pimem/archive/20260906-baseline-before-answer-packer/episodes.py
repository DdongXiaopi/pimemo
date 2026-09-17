from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pimem.core.models import Observation


@dataclass(frozen=True)
class Turn:
    turn_id: str
    observation_id: str
    episode_id: str
    turn_index: int
    role: str
    content: str
    observed_at: str
    source_ref: str

    def as_dict(self) -> Dict[str, Any]:
        return {"turn_id": self.turn_id, "observation_id": self.observation_id,
                "episode_id": self.episode_id, "turn_index": self.turn_index,
                "role": self.role, "content": self.content, "observed_at": self.observed_at,
                "source_ref": self.source_ref}


@dataclass(frozen=True)
class Episode:
    episode_id: str
    root_session_id: str
    scope: Dict[str, Any]
    turn_ids: List[str]
    start_turn: Optional[int]
    end_turn: Optional[int]
    confidence: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {"episode_id": self.episode_id, "root_session_id": self.root_session_id,
                "scope": self.scope, "turn_ids": self.turn_ids, "start_turn": self.start_turn,
                "end_turn": self.end_turn, "confidence": self.confidence}


def build_episode(observations: List[Observation], session_id: str) -> tuple[Episode, List[Turn]]:
    ordered = sorted(observations, key=lambda item: (
        int((item.metadata or {}).get("turn_index", 0)), item.observed_at, item.id))
    episode_id = f"episode:{session_id}"
    turns = [Turn(turn_id=f"turn:{observation.id}", observation_id=observation.id,
                  episode_id=episode_id, turn_index=int((observation.metadata or {}).get("turn_index", index)),
                  role=str((observation.metadata or {}).get("role") or "unknown"), content=observation.content,
                  observed_at=observation.observed_at, source_ref=observation.source_ref)
             for index, observation in enumerate(ordered)]
    indexes = [turn.turn_index for turn in turns]
    episode = Episode(episode_id=episode_id, root_session_id=session_id,
                      scope=ordered[0].scope.as_dict() if ordered else {},
                      turn_ids=[turn.turn_id for turn in turns], start_turn=min(indexes) if indexes else None,
                      end_turn=max(indexes) if indexes else None)
    return episode, turns


def relation_type(previous: Optional[Turn], current: Turn) -> str:
    if previous is None:
        return "precedes"
    if previous.role == "user" and current.role == "assistant":
        return "responds_to"
    if previous.role == current.role:
        return "continues"
    return "precedes"
