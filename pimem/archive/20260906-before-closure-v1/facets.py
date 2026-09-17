from __future__ import annotations

import re
from pimem.core.models import stable_hash
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from pimem.core.models import Facet, Observation


@dataclass
class ClaimProposal:
    subject: str
    predicate: str
    object: Any
    confidence: float
    source_observation_id: str
    source_span: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    facets: List[Facet] = field(default_factory=list)
    claim_proposals: List[ClaimProposal] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    extractor_version: str = "builtin-1"


class FacetExtractor(Protocol):
    def extract(self, observation: Observation) -> ExtractionResult:
        ...


class BuiltinFacetExtractor:
    name = "builtin"
    version = "1"

    def extract(self, observation: Observation) -> ExtractionResult:
        text = observation.content
        facets: List[Facet] = []
        def add_facet(kind: str, raw: Any, normalized: Any, start: int, end: int,
                      confidence: float = 0.9, polarity: Optional[str] = "affirmed") -> None:
            facet_id = "facet_" + stable_hash({"observation": observation.id, "extractor": self.name,
                                                "version": self.version, "kind": kind, "span": [start, end],
                                                "value": normalized})[:24]
            facets.append(Facet(facet_id, kind, raw, normalized, observation.id, start, end,
                                confidence, polarity, "asserted", None, self.name, self.version))
        for match in re.finditer(r"[$€£]?\d+(?:[.,]\d+)?(?:\s*(?:days?|weeks?|months?|years?|hours?|美元|元|天|周|月|年))?", text, re.IGNORECASE):
            raw = match.group(0)
            normalized = raw.replace(",", "")
            kind = "duration" if re.search(r"days?|weeks?|months?|years?|hours?|天|周|月|年", raw, re.IGNORECASE) else "numeric_value"
            add_facet(kind, raw, normalized, match.start(), match.end())
        for match in re.finditer(r"\b(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)\b|周[一二三四五六日天]", text, re.IGNORECASE):
            add_facet("date", match.group(0), match.group(0).lower(), match.start(), match.end(), 0.82)
        for match in re.finditer(r"(?:^|\n)\s*(\d+)[.)]\s+", text):
            add_facet("ordinal", match.group(1), int(match.group(1)), match.start(1), match.end(1), 0.88)
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", text):
            value = match.group(0)
            if value.lower() not in {"The", "This", "Note", "Chapter", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}:
                add_facet("named_entity", value, value, match.start(), match.end(), 0.62)
        lowered = text.lower()
        for phrase, polarity in (("prefer", "positive"), ("like", "positive"), ("love", "positive"),
                                 ("dislike", "negative"), ("hate", "negative"), ("喜欢", "positive"),
                                 ("不喜欢", "negative"), ("偏好", "positive")):
            start = lowered.find(phrase.lower())
            if start >= 0:
                add_facet("speech_act", phrase, phrase, start, start + len(phrase), 0.86, polarity)
        proposals: List[ClaimProposal] = []
        if any(facet.kind == "speech_act" and facet.raw_value.lower() in {"prefer", "like", "love", "hate", "dislike", "喜欢", "不喜欢", "偏好"}
               for facet in facets):
            proposals.append(ClaimProposal("self", "user_preference", text, 0.86, observation.id))
        return ExtractionResult(facets=facets, claim_proposals=proposals, extractor_version=self.version)
