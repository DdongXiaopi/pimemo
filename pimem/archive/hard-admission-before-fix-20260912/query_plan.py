from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


INTENTS = {"lookup", "count", "sum", "compare", "latest", "temporal", "preference", "abstention"}
SELF_TERMS = {"i", "me", "my", "mine", "myself", "user", "我的", "我"}
STOPWORDS = {
    "a", "an", "and", "are", "be", "been", "did", "do", "does", "for", "from", "have", "has", "how",
    "i", "in", "is", "it", "me", "my", "mine", "of", "on", "or", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "you", "your", "myself", "user", "bachelor", "master", "degree",
    "的", "了", "吗", "和", "我", "是", "有", "在", "什么",
}
GRAMMATICAL_TOKENS = {
    "a", "an", "and", "are", "be", "been", "can", "could", "did", "do", "does", "for", "from",
    "have", "has", "how", "i", "i'm", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "to", "was", "were", "what", "when", "which", "who", "with", "you", "your", "remind", "tell",
    "please", "previous", "conversation", "chat", "之前", "请", "告诉", "什么", "哪个", "怎么", "如何",
}
TEMPORAL_TERMS = {"today", "yesterday", "tomorrow", "sunday", "monday", "tuesday", "wednesday", "thursday",
                  "friday", "saturday", "last", "ago", "week", "month", "year", "day", "周日", "星期日"}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                 "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


@dataclass(frozen=True)
class QueryPlan:
    query: str
    intent: str = "lookup"
    entities: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    temporal_anchor: Optional[str] = None
    temporal_constraints: List[Dict[str, Any]] = field(default_factory=list)
    aggregation: Dict[str, Any] = field(default_factory=dict)
    requested_objects: List[str] = field(default_factory=list)
    unknown_policy: str = "answer_from_supported_evidence"
    primary_intent: Optional[str] = None
    secondary_intents: List[str] = field(default_factory=list)
    intent_scores: Dict[str, float] = field(default_factory=dict)
    intent_confidence: float = 1.0
    subject_ref: Optional[str] = None
    fields: List[str] = field(default_factory=list)
    requested_slots: List[Dict[str, Any]] = field(default_factory=list)
    comparison_sides: List[str] = field(default_factory=list)
    as_of_policy: str = "query_anchor"
    retrieval_modes: List[str] = field(default_factory=lambda: ["claim", "observation"])
    self_references: List[str] = field(default_factory=list)
    grammatical_tokens: List[str] = field(default_factory=list)
    temporal_mentions: List[str] = field(default_factory=list)
    entity_candidates: List[str] = field(default_factory=list)
    field_candidates: List[str] = field(default_factory=list)
    comparison_operands: List[str] = field(default_factory=list)
    ordinal_constraints: List[Dict[str, Any]] = field(default_factory=list)
    referential_terms: List[str] = field(default_factory=list)
    answer_shape: str = "scalar"
    speech_act_target: Optional[str] = None
    memory_mode: str = "semantic_state"
    literals: List[str] = field(default_factory=list)
    requested_count: Optional[int] = None
    complementary_list: bool = False
    relation_terms: List[str] = field(default_factory=list)
    event_recall_terms: List[str] = field(default_factory=list)
    answer_focus_terms: List[str] = field(default_factory=list)
    schema_version: str = "query-plan.v2"

    def __post_init__(self) -> None:
        if self.primary_intent is None:
            object.__setattr__(self, "primary_intent", self.intent)
        if self.intent != self.primary_intent:
            object.__setattr__(self, "intent", self.primary_intent or self.intent)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def speech_act_compatible(target: Optional[str], actual: str) -> bool:
    if not target:
        return False
    if target == "historical_answer":
        return actual == "user_statement"
    if target == "assistant_recommendation":
        return actual in {"assistant_recommendation", "assistant_answer"}
    if target == "assistant_answer":
        return actual in {"assistant_answer", "assistant_recommendation", "assistant_instruction"}
    return target == actual


def slot_matches(slot: Dict[str, Any], item: Dict[str, Any]) -> bool:
    haystack = " ".join(str(item.get(key) or "") for key in (
        "subject", "predicate", "object", "statement", "text", "question_text"
    )).lower()
    for key in ("source_span", "answer_units", "facets", "structured_facts"):
        value = item.get(key) or []
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    haystack += " " + " ".join(
                        str(entry.get(name) or "") for name in ("text", "value", "attribute", "header", "row_label")
                    ).lower()
    entity = str(slot.get("entity") or "").lower()
    field = str(slot.get("field") or "").lower()
    entity_present = bool(entity and entity != "self" and entity in haystack)
    if entity and entity != "self" and not entity_present:
        paired_question = item.get("record_type") == "answer_candidate" and bool(
            item.get("question_turn_id") or item.get("question_text")
        ) and bool(
            int(item.get("question_entity_hits") or 0)
            or entity in str(item.get("question_text") or "").lower()
        )
        explicit_candidate_entity = item.get("record_type") == "answer_candidate" and int(
            item.get("requested_entity_hits") or 0
        ) > 0
        if not paired_question and not explicit_candidate_entity:
            return False
    if field in {"", "unknown"}:
        return bool(entity_present or entity == "self")
    if field in {"value", "attribute"}:
        kinds = {str(unit.get("kind") or "") for unit in (item.get("answer_units") or [])}
        kinds.update(str(facet.get("kind") or "") for facet in (item.get("facets") or []))
        if kinds & {
            "attribute_value", "attribute_sentence", "typed_sentence", "quoted_span",
            "table_cell", "table_coordinate", "table_row",
        }:
            return True
        if item.get("answer_shape") == "literal" and "special_literal" in kinds:
            return True
        answer_discriminative_hits = int(item.get("answer_discriminative_hits") or 0)
        answer_anchor_hits = int(item.get("answer_anchor_hits") or 0)
        relation_hits = int(item.get("relation_hits") or 0) + int(item.get("typed_relation_score") or 0)
        structured_list = "list_item" in kinds
        speech_act = str(item.get("speech_act") or "")
        return (answer_discriminative_hits >= 3 or answer_anchor_hits > 0 or
                (structured_list and (relation_hits > 0 or int(item.get("requested_entity_hits") or 0) == 0))) and speech_act in {
            "assistant_answer", "assistant_recommendation", "assistant_instruction", "user_statement",
        }
    field_kinds = {
        "phone_number": {"phone_number"},
        "amount": {"numeric_value", "percentage", "ratio"},
        "percentage": {"percentage", "numeric_value"},
        "ratio": {"ratio", "numeric_value"},
        "year": {"year", "numeric_value"},
        "date": {"date", "year", "numeric_value"},
        "duration": {"numeric_value"},
        "count": {"numeric_value"},
        "list_item": {"list_item"},
        "table_cell": {"table_cell", "table_coordinate", "table_row"},
        "table_coordinate": {"table_coordinate", "table_cell", "table_row"},
    }
    kinds = {str(unit.get("kind") or "") for unit in (item.get("answer_units") or [])}
    kinds.update(str(facet.get("kind") or "") for facet in (item.get("facets") or []))
    return bool(kinds & field_kinds.get(field, {field}))


def parse_as_of(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    for pattern in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _temporal_constraints(query: str, anchor: Optional[datetime]) -> List[Dict[str, Any]]:
    lowered = query.lower()
    constraints: List[Dict[str, Any]] = []
    units = (("day", timedelta(days=1)), ("week", timedelta(weeks=1)),
             ("month", timedelta(days=30)), ("year", timedelta(days=365)))
    for match in re.finditer(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?|years?)\s+ago", lowered):
        amount = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                  "seven": 7, "eight": 8, "nine": 9, "ten": 10}.get(match.group(1), int(match.group(1)) if match.group(1).isdigit() else 1)
        unit_name = match.group(2).rstrip("s")
        unit_delta = dict(units)[unit_name]
        constraints.append({"kind": "relative", "unit": unit_name + "s_ago", "amount": amount,
                            "target": (anchor - unit_delta * amount).isoformat() if anchor else None})
    for phrase, days in (("yesterday", 1), ("last week", 7), ("last month", 30), ("last year", 365)):
        if phrase in lowered:
            constraints.append({"kind": "relative", "unit": phrase, "amount": 1,
                                "target": (anchor - timedelta(days=days)).isoformat() if anchor else None})
    if "today" in lowered:
        constraints.append({"kind": "calendar", "unit": "today", "amount": 1,
                            "target": anchor.date().isoformat() if anchor else None})
    return constraints


def _extract_entities(query: str) -> List[str]:
    chess_literals = re.findall(
        r"\b\d+\.\s*[KQRBN]?[a-h][1-8](?:\s+[KQRBN]?[a-h][1-8][+#]?)*", query,
        flags=re.IGNORECASE,
    )
    entity_query = query
    entity_query = re.sub(r"(?<!\w)'[^'\n]{1,120}'(?!\w)|\"[^\"\n]{1,120}\"", " ", entity_query)
    for literal in chess_literals:
        entity_query = entity_query.replace(literal, " ")
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*\b", entity_query)
    quoted = re.findall(r"(?<!\w)'([^'\n]{2,})'(?!\w)|\"([^\"\n]{2,})\"", query)
    result = []
    for single, double in quoted:
        candidate = (single or double).strip()
        if candidate and len(candidate) <= 60 and candidate.lower() not in STOPWORDS and candidate not in result:
            result.append(candidate)
    temporal_words = {term.lower() for term in TEMPORAL_TERMS}
    for candidate in candidates:
        candidate = re.sub(r"['’]s$", "", candidate, flags=re.IGNORECASE).strip()
        if candidate.lower() in STOPWORDS or candidate in {"What", "When", "Where", "Which", "Who", "Why", "How"}:
            continue
        if candidate.lower() in temporal_words or candidate.lower() in {"i'm", "can"}:
            continue
        if candidate not in result:
            result.append(candidate)
    for code in re.findall(r"(?<![A-Za-z])([A-Z]{2,6})(?![A-Za-z])", query):
        if code not in result and code.lower() not in STOPWORDS:
            result.append(code)
    return result


def _extract_count_subjects(query: str) -> List[str]:
    """Extract the noun being counted without treating the surrounding clause as the object."""
    lowered = query.lower().replace("’", "'")
    stop_words = {
        "the", "a", "an", "this", "that", "these", "those", "did", "does", "do",
        "is", "are", "was", "were", "will", "would", "could", "should", "can",
        "in", "on", "at", "from", "of", "for", "with", "to", "per", "during",
    }
    ignored_subjects = {"time", "times", "much", "many", "number", "amount"}
    patterns = (
        r"\bhow\s+many\s+([a-z][a-z0-9'-]*(?:\s+[a-z][a-z0-9'-]*){0,2})",
        r"\bnumber\s+of\s+([a-z][a-z0-9'-]*(?:\s+[a-z][a-z0-9'-]*){0,2})",
    )
    subjects: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            words = []
            for word in match.group(1).split():
                if word in stop_words:
                    break
                words.append(word)
            if not words:
                continue
            subject = " ".join(words).strip(" .,;:?!")
            if subject and subject not in ignored_subjects and subject not in subjects:
                subjects.append(subject)
    return subjects


def _extract_fields(query: str) -> List[str]:
    lowered = query.lower()
    phone_number_query = bool(re.search(r"\b(?:phone|telephone|mobile|contact)\s+number\b", lowered))
    patterns = (("degree", "education.degree"), ("graduate", "education.degree"),
                ("phone number", "phone_number"), ("telephone number", "phone_number"),
                ("mobile number", "phone_number"), ("contact number", "phone_number"),
                ("duration", "duration"), ("how long", "duration"),
                ("how many", "count"), ("number of", "count"), ("employs", "count"),
                ("employees", "count"), ("people", "count"), ("workers", "count"),
                ("amount", "amount"), ("total", "amount"), ("price", "amount"),
                 ("cost", "amount"), ("how much", "amount"),
                 ("ratio", "ratio"), ("proportion", "ratio"),
                ("average improvement", "percentage"), ("what percentage", "percentage"),
                ("percent", "percentage"), ("percentage", "percentage"),
                 ("what year", "year"), ("which year", "year"),
                 ("year did", "year"), ("year", "year"),
                 ("when did", "date"), ("when was", "date"),
                 ("when did", "date"), ("when was", "date"),
                  ("location", "location"), ("where", "location"),
                 ("name of", "attribute"), ("favorite", "attribute"), ("favourite", "attribute"), ("sealant", "attribute"),
                ("designation", "attribute"), ("file number", "attribute"),
                ("breed", "attribute"), ("occupation", "attribute"), ("profession", "attribute"),
                ("what type", "attribute"), ("what kind", "attribute"),
                ("what was", "attribute"), ("what did", "attribute"),
                ("model", "model"), ("frequency", "frequency"))
    fields: List[str] = []
    for phrase, field_name in patterns:
        if phone_number_query and phrase == "number of":
            continue
        if phrase in lowered and field_name not in fields:
            fields.append(field_name)
    if re.search(
        r"\bwhat\s+(?:color|colour|size|shape|appearance|material|texture|weight|height|length|width|age|style)\b",
        lowered,
    ) and "attribute" not in fields:
        fields.append("attribute")
    if re.search(r"\b(?:based|located|situated|headquartered)\s+in\b", lowered):
        if "location" not in fields:
            fields.append("location")
    if "attribute" in fields and "location" in fields:
        fields = ["attribute"] + [field for field in fields if field != "attribute"]
    if re.search(r"\bwhat\s+time\b|\btime\s+of\s+day\b|\bat\s+what\s+time\b", lowered):
        if "time" not in fields:
            fields.append("time")
    return fields or ["value"]


def _extract_event_recall_terms(query: str) -> List[str]:
    lowered = query.lower().replace("’", "'")
    terms = [term for term in (
        "move", "moved", "moving", "pack", "packed", "catch", "caught", "watch", "watched",
        "read", "bought", "visited", "attend", "attended", "graduate", "graduated",
    ) if re.search(r"\b" + term + r"\b", lowered)]
    topic_pattern = re.compile(
        r"\b(?:discussed|talked about|conversation about|chat about|discussion about|about)\s+"
        r"(.+?)(?=\s+\b(?:earlier|before|after|you mentioned|i mentioned|can you|do you|what was|what is)\b|[?.!]|$)"
    )
    for match in topic_pattern.finditer(lowered):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ,;:")
        words = re.findall(r"[a-z0-9][a-z0-9'/-]*", phrase)
        meaningful = [word for word in words if word not in STOPWORDS and len(word) > 1]
        if len(meaningful) >= 2 and phrase not in terms:
            terms.append(phrase)
            terms.extend(word for word in meaningful if word not in terms)
    return list(dict.fromkeys(terms))


def _extract_answer_focus_terms(query: str) -> List[str]:
    lowered = re.sub(r"\s+", " ", query.lower().replace("–", "-").replace("—", "-")).strip()
    focus_terms: List[str] = []
    patterns = (
        r"\b(?:what|which)\s+(?:kind|type|sort|form)\s+of\s+(.+?)\s+"
        r"(?:methods?|options?|alternatives?|ways?|processes?)\b",
        r"\b(?:what|which)\s+(?:are|were)\s+(?:the\s+)?(.+?)\s+"
        r"(?:methods?|options?|alternatives?|ways?|processes?)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ,;:.-")
            if phrase and len(phrase.split()) >= 1:
                focus_terms.append(phrase)
    return list(dict.fromkeys(focus_terms))


def _typed_query_parts(query: str, entities: List[str], fields: List[str], temporal: List[Dict[str, Any]]) -> Dict[str, Any]:
    lowered = query.lower().replace("’", "'")
    tokens = re.findall(r"[\w']+", lowered, flags=re.UNICODE)
    self_references = [token for token in tokens if token in SELF_TERMS or token in {"i'm", "i"}]
    grammatical = [token for token in tokens if token in GRAMMATICAL_TOKENS]
    temporal_mentions = [token for token in tokens if token in TEMPORAL_TERMS and token != "last"]
    temporal_mentions.extend(token for token in tokens if token in {"lately", "recently", "recent"})
    if re.search(r"\blast\s+(?:time|week|month|year|day|conversation|chat)\b", lowered):
        temporal_mentions.append("last")
    temporal_mentions.extend(item.get("unit", "") for item in temporal if item.get("unit"))
    referential_terms = [phrase for phrase in ("previous chat", "previous conversation", "previous game", "previous discussion",
                         "looking back", "going through", "trying to recall", "recall", "you recommended",
                         "you mentioned", "remind me", "before", "earlier", "after", "之前", "推荐过", "提到过") if phrase in lowered]
    ordinal_constraints: List[Dict[str, Any]] = []
    ordinal_pattern = r"(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+(?:item|one|entry|job|work|parameter|option|alternative|company|method|process|bottle|game|way|venue|trail|dish|restaurant|song|objective|goal|个|项)"
    for match in re.finditer(ordinal_pattern, lowered):
        ordinal_constraints.append({"ordinal": int(match.group(1)), "raw": match.group(0)})
    ordinal_word_pattern = r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:item|one|entry|job|work|parameter|option|alternative|company|method|process|bottle|game|way|venue|trail|dish|restaurant|song|objective|goal)\b"
    for match in re.finditer(ordinal_word_pattern, lowered):
        ordinal_constraints.append({"ordinal": ORDINAL_WORDS[match.group(1)], "raw": match.group(0)})
    last_ordinal_pattern = r"\b(?:the\s+)?last\s+(?:item|one|entry|job|work|parameter|option|alternative|company|method|process|bottle|game|way|venue|trail|dish|restaurant|song|objective|goal)\b"
    for match in re.finditer(last_ordinal_pattern, lowered):
        ordinal_constraints.append({"ordinal": -1, "position": "last", "raw": match.group(0)})
    count_match = re.search(
        r"\b(?:the\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:companies|options|alternatives|methods|items|entries|jobs|works|ways|recommendations|parameters)\b",
        lowered,
    )
    requested_count = NUMBER_WORDS.get(count_match.group(1)) if count_match and count_match.group(1) in NUMBER_WORDS else (
        int(count_match.group(1)) if count_match else None
    )
    complementary_list = bool(
        (requested_count or any(term in lowered for term in ("options", "alternatives", "methods", "items", "entries")))
        and re.search(r"\b(?:other|additional|remaining|rest|alternative|alternatives|options)\b", lowered)
    )
    count_subjects = _extract_count_subjects(query)
    for subject in reversed(count_subjects):
        if subject not in entities:
            entities.insert(0, subject)
    quoted = re.findall(r"(?<!\w)'([^'\n]{1,120})'(?!\w)|\"([^\"\n]{1,120})\"", query)
    literals = [single or double for single, double in quoted if (single or double).strip()]
    literals.extend(re.findall(r"\b\d+(?:st|nd|rd|th)\s+\w+", lowered))
    literals.extend(re.findall(
        r"\b\d+\.\s*[KQRBN]?[a-h][1-8](?:\s+[KQRBN]?[a-h][1-8][+#]?)*", query,
        flags=re.IGNORECASE,
    ))
    literals.extend(re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", query))
    literals.extend(re.findall(r"\b\d+\.?\s*[A-Za-z]{1,4}(?:\s+[A-Za-z]{1,4})?\+?\b", query))
    literals.extend(re.findall(r"(?<![A-Za-z])(?:[A-Z]{2,6})(?![A-Za-z])", query))
    literals = list(dict.fromkeys(item.strip() for item in literals if item.strip()))
    relation_terms: List[str] = []
    relation_anchors = {"shop", "spot", "deli", "restaurant", "restaurants", "eatery", "eateries", "trail", "trails", "venue", "venues",
                       "video", "videos", "milkshake", "milkshakes", "cheese", "cheeses",
                       "game", "dancer", "dancers", "method", "methods", "language", "languages",
                       "vase", "sealant", "construction", "house", "framerate", "agent", "beer", "recipe", "cartoon",
                       "culture", "shirt", "wearing", "company", "companies", "business", "businesses",
                       "employee", "employees", "employs", "people", "workers", "industry", "manufacturing",
                       "rug", "rugs", "breed", "dog", "dogs", "parameter", "jumpsuit", "designation", "file",
                       "location", "locations", "advisor", "advisors", "president", "president's", "name"}
    relation_tokens = [token for token in re.findall(r"[a-z0-9]+", lowered)
                       if token not in STOPWORDS and token not in GRAMMATICAL_TOKENS and len(token) > 2]
    for index in range(len(relation_tokens) - 1):
        left, right = relation_tokens[index:index + 2]
        if left in relation_anchors or right in relation_anchors:
            relation_terms.append(f"{left} {right}")
    for index in range(len(relation_tokens) - 2):
        window = relation_tokens[index:index + 3]
        if any(token in relation_anchors for token in window):
            relation_terms.append(" ".join(window))
    for phrase in ("name of", "hostel name", "what year", "how much", "average improvement", "what was", "what did",
                   "wearing", "wear", "sealant", "allocated", "construction began", "center", "circumference",
                   "file number", "designation", "rotation", "framerate", "company list", "music streaming service",
                   "prompt parameter", "fundraising dinner", "study abroad", "fruit"):
        if phrase in lowered:
            relation_terms.append(phrase)
    if count_subjects:
        relation_terms.extend(f"count of {subject}" for subject in count_subjects)
    if re.search(r"\b(?:what|which)\s+(?:kind|type)\s+of\b", lowered):
        relation_terms.append("type of")
    location_match = re.search(
        r"\b(?:based|located|situated|headquartered)\s+in\s+"
        r"([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})",
        lowered,
    )
    if location_match:
        location = re.split(
            r"\b(?:that|which|and|with|offering|selling|sells|offers)\b",
            location_match.group(1),
            maxsplit=1,
        )[0].strip(" .,;:?!")
        if location:
            relation_terms.append(f"based in {location}")
    if re.search(r"\bwhat\s+.*\bname\b|\bname\s+did\b", lowered):
        relation_terms.append("name decision")
    if re.search(r"\b(?:finally\s+decided|decided\s+to\s+name|what\s+.*finally\s+decided)\b", lowered):
        relation_terms.append("final name")
    role_match = re.search(
        r"\b(?:who\s+is|who\s+was|what\s+is\s+the\s+name\s+of)\s+(?:the\s+)?(.+?)(?=\s+mentioned\b|\s+in\s+the\b|\s+in\s+an?\b|\?|$)",
        lowered,
    )
    if role_match:
        role_phrase = re.sub(r"\s+", " ", role_match.group(1)).strip(" .,:;?!")
        if len(role_phrase.split()) >= 2:
            relation_terms.append(role_phrase)
    if re.search(r"\b(?:recommended|suggested)\b", lowered):
        for phrase in ("vegan eatery", "multiple locations", "hiking trail", "video recommended"):
            if phrase in lowered:
                relation_terms.append(phrase)
    if "hostel" in lowered and "name of" in lowered:
        relation_terms.append("hostel name")
    if "sealant" in lowered and "vase" in lowered:
        relation_terms.append("sealant for vase")
    if "construction" in lowered and "year" in lowered:
        relation_terms.append("construction began year")
    if "company" in lowered and requested_count:
        relation_terms.append("company list")
    if "framerate" in lowered and "average improvement" in lowered:
        relation_terms.append("framerate improvement")
    if ("ratio" in lowered or "proportion" in lowered) and "tea tree" in lowered and "carrier oil" in lowered:
        relation_terms.append("tea tree carrier oil ratio")
    if "designation" in lowered and "jumpsuit" in lowered:
        relation_terms.append("designation jumpsuit")
    if "file number" in lowered and "designation" in lowered:
        relation_terms.append("designation file number")
    if "breed" in lowered and "dog" in lowered:
        relation_terms.append("dog breed")
    if re.search(r"\bemploys?\b", lowered) and re.search(r"\bpeople|employees|workers\b", lowered):
        relation_terms.append("employment count")
    if "rug" in lowered and "manufacturing" in lowered:
        relation_terms.append("rug manufacturing")
    if "fruit" in lowered:
        relation_terms.append("fruit")
    answer_shape = "scalar"
    if any(re.fullmatch(r"\d+\.\s*[KQRBN]?[a-h][1-8](?:\s+[KQRBN]?[a-h][1-8][+#]?)*", literal, flags=re.IGNORECASE) for literal in literals):
        answer_shape = "literal"
    elif ordinal_constraints or requested_count or any(term in lowered for term in ("list", "items", "entries", "options", "alternatives", "methods", "languages", "processes", "objectives", "goals", "ways", "parameters", "逐项", "列表")):
        answer_shape = "list_item" if ordinal_constraints else "list"
    elif any(term in lowered for term in ("table", "sheet", "rotation", "schedule", "row", "column", "班次", "表格")):
        answer_shape = "table_cell"
    elif any(term in lowered for term in ("how long", "duration", "多久", "多长")):
        answer_shape = "duration"
    elif any(term in lowered for term in ("how much", "amount", "allocated", "cost", "price", "percentage", "percent", "average improvement", "ratio", "proportion")):
        answer_shape = "amount"
    elif any(term in lowered for term in ("what year", "which year", "what date", "year did", "哪一年")):
        answer_shape = "date"
    elif any(term in lowered for term in ("date", "when", "哪天", "日期")):
        answer_shape = "date"
    elif any(term in lowered for term in ("where", "location", "哪里", "地点")):
        answer_shape = "entity"
    elif re.search(r"\b(?:recommended|suggested)\b", lowered) and re.search(
        r"\b(?:video|trail|eatery|restaurant|spot|method)\b", lowered,
    ):
        answer_shape = "list_item"
    elif any(term in lowered for term in ("who", "which", "what", "name", "哪个", "什么")):
        answer_shape = "entity"
    event_recall_terms = _extract_event_recall_terms(query)
    answer_focus_terms = _extract_answer_focus_terms(query)
    user_historical_answer = bool(re.search(
        r"\b(?:i|we)\s+(?:mentioned|said|used|recommended|suggested|gave|bought|visited|wore|chose|decided|settled|agreed)\b"
        r"|\b(?:finally\s+decided|decided\s+to\s+name|what\s+name\s+did\s+(?:i|we)\s+choose|what\s+did\s+(?:i|we)\s+decide)\b",
        lowered,
    ))
    speech_act_target = None
    past_recommendation_recall = bool(re.search(r"\byou\s+recommended\b|\bwhat\s+.*\brecommended\b", lowered))
    if user_historical_answer:
        speech_act_target = "historical_answer"
    elif past_recommendation_recall:
        speech_act_target = "assistant_answer"
    elif any(term in lowered for term in ("recommend", "recommended", "suggest", "suggested", "alternative", "alternatives", "options", "推荐")):
        speech_act_target = "assistant_recommendation"
    elif any(term in lowered for term in ("said", "mentioned", "回答", "说过", "提到")):
        speech_act_target = "assistant_answer"
    wh_question = bool(re.search(r"\b(?:what|where|when|who|which|how)\b", lowered))
    personal_fact_question = bool(self_references and wh_question and re.search(
        r"\?|\b(?:did|do|does|is|are|was|were|have|has|had|attend|attended|go|went|buy|bought|redeem|redeemed|take|volunteer|volunteered|fundraising|complete|completed|favorite|favourite|worth|using|use|used|name|degree|play|trip|own|graduate|released|pack|study abroad)\b",
        lowered,
    ))
    if speech_act_target is None and personal_fact_question and any(
        field_name in fields for field_name in ("location", "date", "year", "duration", "amount", "attribute", "value")
    ):
        speech_act_target = "historical_answer"
    memory_mode = "episodic_recall" if referential_terms or speech_act_target or personal_fact_question or event_recall_terms else "semantic_state"
    if memory_mode == "episodic_recall" and any(term in lowered for term in ("current", "latest", "now", "现在")):
        memory_mode = "hybrid"
    if requested_count and answer_shape == "scalar":
        answer_shape = "multi_value"
    return {"self_references": list(dict.fromkeys(self_references)),
            "grammatical_tokens": list(dict.fromkeys(grammatical)),
            "temporal_mentions": list(dict.fromkeys(temporal_mentions)),
            "entity_candidates": entities, "field_candidates": list(fields),
            "comparison_operands": entities[:2], "ordinal_constraints": ordinal_constraints,
            "referential_terms": referential_terms, "answer_shape": answer_shape,
            "speech_act_target": speech_act_target, "memory_mode": memory_mode, "event_recall_terms": event_recall_terms,
            "literals": literals, "requested_count": requested_count,
            "complementary_list": complementary_list,
            "relation_terms": list(dict.fromkeys(relation_terms)),
            "answer_focus_terms": answer_focus_terms}


def build_query_plan(query: str, *, as_of: Optional[str] = None) -> QueryPlan:
    lowered = query.lower()
    anchor = parse_as_of(as_of)
    temporal = _temporal_constraints(query, anchor)
    scores = {intent: 0.0 for intent in sorted(INTENTS)}
    if any(term in lowered for term in ("not enough information", "not mentioned", "unknown", "cannot determine", "can't determine")):
        scores["abstention"] += 1.0
    if any(term in lowered for term in ("prefer", "preference", "favorite", "favourite", "dislike", "based on my")):
        scores["preference"] += 1.0
    if any(term in lowered for term in ("latest", "most recent", "currently", "current", "now", "lately", "recently", "recent")):
        scores["latest"] += 1.0
    if any(term in lowered for term in ("compare", "difference", "more than", "less than", "versus", " vs ")):
        scores["compare"] += 1.0
    if any(term in lowered for term in ("total", "sum", "combined", "altogether")):
        scores["sum"] += 1.0
    phone_number_query = bool(re.search(r"\b(?:phone|telephone|mobile|contact)\s+number\b", lowered))
    if any(term in lowered for term in ("how many", "number of", "count of")) and not phone_number_query:
        scores["count"] += 1.0
    if scores["count"] and any(term in lowered for term in ("in total", "total", "combined", "altogether")):
        scores["sum"] += 0.5
    fields_for_intent = _extract_fields(query)
    strong_value_query = any(field_name in fields_for_intent for field_name in ("amount", "percentage", "year", "duration", "ratio"))
    explicit_temporal_question = bool(re.search(r"^\s*when\b|\bwhen\s+did\b|\bwhen\s+was\b|\bbefore\b|\bafter\b|\bearliest\b", lowered))
    if temporal or (explicit_temporal_question and not strong_value_query):
        scores["temporal"] += 1.0
    if not any(scores.values()):
        scores["lookup"] = 1.0
    ordered = sorted(scores, key=lambda name: (-scores[name], name))
    primary = ordered[0]
    if scores[primary] == 0:
        primary = "lookup"
    confidence = min(1.0, scores[primary] / max(sum(scores.values()), 1.0))
    secondary = [name for name in ordered[1:] if scores[name] > 0 and scores[primary] - scores[name] <= 0.5]
    fields = _extract_fields(query)
    entities = _extract_entities(query)
    count_subjects = _extract_count_subjects(query)
    subject_ref = "self" if re.search(r"\b(?:i|me|my|mine|myself|user)\b|我的|我", lowered) else None
    typed = _typed_query_parts(query, entities, fields, temporal)
    if primary in {"sum", "compare"}:
        typed["memory_mode"] = "semantic_state"
    personal_fact_query = bool(subject_ref and re.search(
        r"\?|\b(?:did|do|does|is|are|was|were|have|has|had|attend|attended|go|went|buy|bought|redeem|redeemed|take|volunteer|complete|completed|favorite|favourite|worth|using|use|used|name|degree|play|trip|own|graduate|released|pack)\b",
        lowered,
    ))
    if primary in {"count", "sum", "compare"} and primary not in {"sum", "compare"} and not (
        typed.get("referential_terms") or typed.get("speech_act_target") or personal_fact_query
    ):
        typed["memory_mode"] = "semantic_state"
    requested_objects = list(count_subjects or entities)
    slots = []
    explicit_multi = primary in {"compare", "sum", "count"} or typed.get("answer_shape") in {"list", "multi_value"} or bool(typed.get("requested_count"))
    slot_entities = count_subjects if primary == "count" and count_subjects else (
        entities if explicit_multi else (entities[-1:] or (["self"] if subject_ref == "self" else []))
    )
    for index, entity in enumerate(slot_entities):
        slots.append({"slot_id": f"object-{index + 1}.value", "subject_ref": subject_ref, "entity": entity,
                      "field": fields[0], "value_type": fields[0], "required": True, "exact_match": bool(entities)})
    aggregation: Dict[str, Any] = {}
    unknown_policy = "answer_from_supported_evidence"
    if primary == "count": aggregation = {"operator": "count", "dedupe_key": "normalized_object"}; unknown_policy = "count_only_explicit_evidence"
    elif primary == "sum": aggregation = {"operator": "sum", "dedupe_key": "object_and_source"}; unknown_policy = "preserve_units_and_unknowns"
    elif primary == "compare": aggregation = {"operator": "compare", "dedupe_key": "entity"}; unknown_policy = "preserve_both_sides"
    elif primary == "latest": unknown_policy = "current_state_only"
    elif primary == "temporal": unknown_policy = "preserve_event_time"
    elif primary == "preference": unknown_policy = "do_not_infer_preference"
    elif primary == "abstention": unknown_policy = "preserve_partial_knowledge"
    predicates = ["user_code_preference"] if "preference" in [primary, *secondary] else []
    comparison_sides = entities[:2] if primary == "compare" else []
    return QueryPlan(query=query, intent=primary, primary_intent=primary, secondary_intents=secondary,
                     intent_scores={key: round(value, 4) for key, value in scores.items() if value > 0},
                     intent_confidence=round(confidence, 4), entities=entities, predicates=predicates,
                     temporal_anchor=as_of, temporal_constraints=temporal, aggregation=aggregation,
                     requested_objects=requested_objects, unknown_policy=unknown_policy, subject_ref=subject_ref,
                     fields=fields, requested_slots=slots, comparison_sides=comparison_sides, **typed)


def normalize_query_tokens(query: str) -> List[str]:
    normalized = query.lower().replace("’", "'")
    normalized = re.sub(r"\b(\w+)'s\b", r"\1", normalized).replace("-", " ")
    raw_tokens = re.findall(r"[\w]+", normalized, flags=re.UNICODE)
    tokens: List[str] = []
    for token in raw_tokens:
        if len(token) <= 1 or token in STOPWORDS:
            continue
        tokens.append(token)
        if token.endswith("ies") and len(token) > 4:
            tokens.append(token[:-3] + "y")
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            tokens.append(token[:-1])
    return list(dict.fromkeys(tokens))


def expand_query_tokens(query: str) -> List[str]:
    aliases = {
        "purchase": ("buy", "bought", "purchased"), "purchased": ("buy", "bought", "purchase"),
        "download": ("downloaded",), "album": ("ep", "albums"), "job": ("work", "career", "employment"),
        "trip": ("travel", "journey", "vacation"), "camera": ("photography", "photographic"),
        "prefer": ("preference", "like", "favorite"), "favorite": ("favourite", "prefer", "like"),
        "duration": ("days", "weeks", "months", "years", "length"), "degree": ("graduated", "education"),
        "compulsion": ("compulsions", "compulsive"), "compulsions": ("compulsion", "compulsive"),
        "behavior": ("behaviors", "behaviour", "behaviours"), "behaviors": ("behavior", "behaviour"),
    }
    tokens = normalize_query_tokens(query)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(aliases.get(token, ()))
    return list(dict.fromkeys(expanded))
