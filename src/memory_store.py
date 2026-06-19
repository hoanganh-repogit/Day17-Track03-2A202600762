from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate token count with a stable character-based heuristic."""
    text = (text or "").strip()
    if not text:
        return 0

    return max(1, (len(text) + 3) // 4)


_FACT_LINE_RE = re.compile(
    r"^\s*-\s+(?:\*\*)?(?P<key>[A-Za-z0-9_-]+)(?:\*\*)?\s*:\s*(?P<value>.*?)\s*$"
)


@dataclass
class UserProfileStore:
    """Persistent markdown storage for user profile facts."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = "".join(
            ch if ch.isascii() and (ch.isalnum() or ch in "_-") else "_"
            for ch in user_id.strip()
        ).strip("_-")
        return self.root_dir / f"{safe_id or 'user'}.md"

    def read_text(self, user_id: str) -> str:
        try:
            return self.path_for(user_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            return "# User Profile\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text or search_text == replacement:
            return False

        path = self.path_for(user_id)
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            content = "# User Profile\n\n"

        index = content.find(search_text)
        if index == -1:
            return False

        updated = content[:index] + replacement + content[index + len(search_text) :]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        try:
            return self.path_for(user_id).stat().st_size
        except FileNotFoundError:
            return 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = _FACT_LINE_RE.match(line)
            if match is None:
                continue

            key = match.group("key").strip().lower()
            value = match.group("value").strip()
            if key and value:
                facts[key] = value
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        safe_key = "".join(
            ch.lower() if ch.isascii() and (ch.isalnum() or ch in "_-") else "_"
            for ch in key.strip()
        ).strip("_-")
        clean_value = _SPACE_RE.sub(" ", value or "").strip()
        if not safe_key or not clean_value:
            return self.path_for(user_id)

        content = self.read_text(user_id)
        lines = content.splitlines()
        if not lines:
            lines = ["# User Profile", ""]

        replacement = f"- {safe_key}: {clean_value}"
        replaced = False
        for index, line in enumerate(lines):
            match = _FACT_LINE_RE.match(line)
            if match is not None and match.group("key").strip().lower() == safe_key:
                lines[index] = replacement
                replaced = True
                break

        if not replaced:
            if lines[-1].strip():
                lines.append("")
            lines.append(replacement)

        return self.write_text(user_id, "\n".join(lines).rstrip() + "\n")


_SPACE_RE = re.compile(r"\s+")
_TRAILING_FILLER_RE = re.compile(
    r"\s*(?:nhé|nha|giúp mình|giúp tôi|cho chắc|please|thanks)\s*\.?$",
    re.IGNORECASE,
)
_QUESTION_WORD_RE = re.compile(
    r"(?:\b(?:what|who|where|which|how)\b|(?:^|\s)(?:gì|đâu|nào)(?:\s|$))",
    re.IGNORECASE,
)
_PROFILE_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "name",
        re.compile(
            r"(?:\b(?:mình|tôi|tớ|em|anh|chị)\s+tên(?:\s+là)?|"
            r"tên\s+(?:của\s+)?(?:mình|tôi|tớ|em|anh|chị)\s+là)\s+"
            r"(?P<value>[^,.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "name",
        re.compile(r"\bmy\s+name\s+is\s+(?P<value>[^,.!?;\n]+)", re.IGNORECASE),
    ),
    (
        "name",
        re.compile(
            r"(?:^|[,:;]\s*)tên\s+(?P<value>[^,.;!?]+)"
            r"(?=,\s*(?:nghề|nơi ở|style|phong cách))",
            re.IGNORECASE,
        ),
    ),
    (
        "location",
        re.compile(
            r"(?:\b(?:mình|tôi|tớ|em|anh|chị)\s+(?:vẫn\s+|hiện\s+|đang\s+)*"
            r"(?:ở|sống\s+ở)|\bhiện\s+ở)\s+(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "location",
        re.compile(
            r"(?:nơi ở|địa điểm|location)(?:\s+hiện tại)?\s*(?:là|:)\s*"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "location",
        re.compile(
            r"(?:thực ra|từ tuần này|giờ|hiện tại)[^.!?;\n]{0,60}?"
            r"\b(?:ở|làm việc ở|live in|am in)\s+(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "location",
        re.compile(
            r"\b(?:i\s+live\s+in|i(?:'m| am)\s+in|currently\s+in)\s+"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "profession",
        re.compile(
            r"(?:\b(?:mình|tôi|tớ|em|anh|chị)\s+(?:vẫn\s+|hiện\s+|đang\s+)*"
            r"làm(?:\s+nghề)?|\b(?:vẫn\s+|hiện\s+|đang\s+|giờ\s+)*đang\s+làm|"
            r"nghề(?:\s+nghiệp)?[^.!?;\n]{0,20}?\s*(?:là|:))\s+"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "profession",
        re.compile(
            r"(?:^|[,:;]\s*)nghề(?:\s+nghiệp)?\s+(?P<value>[^,.;!?]+)"
            r"(?=,\s*(?:nơi ở|style|phong cách)|[.;!?]?$)",
            re.IGNORECASE,
        ),
    ),
    (
        "profession",
        re.compile(
            r"(?:giờ\s+)?(?:chuyển\s+sang|trở thành)\s+(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "profession",
        re.compile(
            r"\b(?:i\s+work\s+as|my\s+(?:job|profession|role)\s+is)\s+"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "profession",
        re.compile(
            r"\bi(?:'m| am)\s+(?:an?\s+)?(?P<value>[^.!?;\n]*"
            r"(?:engineer|developer|designer|manager|student|teacher|researcher|"
            r"analyst|consultant|scientist|founder|architect|writer|doctor|nurse|"
            r"lawyer|professor|giáo viên|sinh viên|bác sĩ|kỹ sư|lập trình viên)"
            r"[^.!?;\n]*)",
            re.IGNORECASE,
        ),
    ),
    (
        "response_style",
        re.compile(
            r"(?:\b(?:mình|tôi|tớ|em|anh|chị)\s+(?:vẫn\s+)?muốn\s+(?:bạn\s+)?"
            r"(?:trả lời|câu trả lời)|\bhãy\s+trả lời|\bkhi\s+giải\s+thích[^,.!?;\n]*"
            r"hãy\s+trả lời)\s+(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "response_style",
        re.compile(
            r"(?:\b(?:mình|tôi|tớ|em|anh|chị)\s+(?:vẫn\s+)?muốn\s+)?"
            r"(?:style|kiểu|phong cách)\s+trả lời"
            r"(?:\s+(?:mình|tôi|tớ|em)\s+thích)?\s+(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "response_style",
        re.compile(
            r"(?:style|kiểu|phong cách)\s+trả lời[^:.\n]{0,50}:\s*"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "response_style",
        re.compile(
            r"\b(?:please\s+)?(?:answer|reply|respond)\s+(?:in|with)?\s*"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "response_style",
        re.compile(
            r"\bi\s+prefer\s+(?:answers?|responses?)\s+(?:that\s+are\s+|to\s+be\s+)?"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "favorite_food",
        re.compile(
            r"(?:món ăn|đồ ăn|food)\s+yêu\s+thích(?:\s+của\s+(?:mình|tôi|tớ|em))?"
            r"\s*(?:là|:)\s*(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "favorite_food",
        re.compile(
            r"\bmy\s+favorite\s+food\s+is\s+(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "favorite_drink",
        re.compile(
            r"(?:đồ uống|thức uống|drink)\s+yêu\s+thích"
            r"(?:\s+của\s+(?:mình|tôi|tớ|em))?\s*(?:là|:)\s*"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "favorite_drink",
        re.compile(
            r"\bmy\s+favorite\s+drink\s+is\s+(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "response_style",
        re.compile(
            r"\b(?:mình|tôi|tớ|em|anh|chị)\s+thích\s+"
            r"(?:cách\s+giải\s+thích|kiểu\s+trả lời|câu\s+trả lời)\s+"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "interests",
        re.compile(
            r"\b(?:mình|tôi|tớ|em|anh|chị)\s+(?:đang\s+)?"
            r"(?:quan\s+tâm(?:\s+nhiều)?\s+(?:đến|tới)|thích)\s+"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "pet",
        re.compile(
            r"\b(?:mình|tôi|tớ|em|anh|chị)\s+nuôi\s+"
            r"(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        ),
    ),
)
_FACT_VALUE_SPLITS = {
    "name": re.compile(r"[,;:]|\s+(?:and\s+i\b|và\b|nhưng\b|chứ\b|dù\b)", re.IGNORECASE),
    "location": re.compile(
        r"[,;]|\s+(?:và\s+(?:đang|hiện|làm)|and\s+i\b|chứ\b|nhưng\b|dù\b|"
        r"còn\b|để\b|vì\b|khi\b|while\b|but\b)",
        re.IGNORECASE,
    ),
    "profession": re.compile(
        r"[,;]|\s+(?:and\s+i\b|chứ\b|nhưng\b|dù\b|còn\b|while\b|but\b)",
        re.IGNORECASE,
    ),
}
_MAX_FACT_WORDS = {
    "name": 6,
    "location": 10,
    "profession": 16,
    "favorite_food": 10,
    "favorite_drink": 10,
    "pet": 12,
}


def _clean_fact_value(key: str, value: str) -> str:
    value = _SPACE_RE.sub(" ", value).strip(" \t\r\n\"'`:-")
    splitter = _FACT_VALUE_SPLITS.get(key)
    if splitter is not None:
        value = splitter.split(value, 1)[0].strip(" \t\r\n\"'`:-")
    value = _TRAILING_FILLER_RE.sub("", value).strip(" \t\r\n\"'`:-")
    return value


def _is_confident_fact(key: str, value: str) -> bool:
    lowered = value.casefold()
    if not value or lowered in {"ai", "gì", "đâu", "nào"} or _QUESTION_WORD_RE.search(value):
        return False

    if key == "profession" and lowered.startswith(("việc ", "viec ", "work ")):
        return False
    if key == "interests" and lowered.startswith(("tin này", "chuyện ", "việc ")):
        return False

    max_words = _MAX_FACT_WORDS.get(key)
    if max_words is not None and len(value.split()) > max_words:
        return False
    return True


def extract_profile_updates(message: str) -> dict[str, str]:
    text = _SPACE_RE.sub(" ", message or "").strip()
    if not text:
        return {}

    lowered = text.casefold()
    assertion_markers = (
        "mình tên",
        "tôi tên",
        "tên mình là",
        "my name is",
        "mình ở",
        "tôi ở",
        "hiện ở",
        "nơi ở",
        "i live in",
        "currently in",
        "đang làm",
        "nghề",
        "i work as",
        "my job is",
        "mình muốn",
        "tôi muốn",
        "hãy trả lời",
        "answer",
        "prefer",
        "yêu thích",
        "quan tâm",
        "mình thích",
        "tôi thích",
        "nuôi",
    )
    if text.endswith("?") and not any(marker in lowered for marker in assertion_markers):
        return {}

    matches: list[tuple[int, str, str]] = []
    for key, pattern in _PROFILE_FACT_PATTERNS:
        for match in pattern.finditer(text):
            value = _clean_fact_value(key, match.group("value"))
            if _is_confident_fact(key, value):
                matches.append((match.start(), key, value))

    updates: dict[str, str] = {}
    for _, key, value in sorted(matches):
        updates[key] = value
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact heuristic summary of older chat messages."""
    if max_items <= 0 or not messages:
        return ""

    selected = messages[-max_items:]
    omitted = len(messages) - len(selected)
    lines: list[str] = []

    if omitted > 0:
        lines.append(f"{omitted} earlier message(s) omitted.")

    for message in selected:
        role = _SPACE_RE.sub(" ", str(message.get("role") or "message")).strip()
        content = _SPACE_RE.sub(" ", str(message.get("content") or "")).strip()
        if not content:
            continue
        if len(content) > 280:
            content = content[:277].rstrip() + "..."
        lines.append(f"- {role}: {content}")

    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """
    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0, "tokens": 0},
        )
        messages = thread["messages"]
        assert isinstance(messages, list)

        message = {"role": role, "content": content}
        messages.append(message)
        thread["tokens"] = int(thread["tokens"]) + self._message_tokens(message)

        keep = max(0, self.keep_messages)
        old_count = len(messages) - keep
        if old_count <= 0 or int(thread["tokens"]) <= max(0, self.threshold_tokens):
            return

        older = messages[:old_count]
        recent = messages[old_count:]
        previous_summary = str(thread["summary"])
        new_summary = summarize_messages(older)
        if previous_summary and new_summary:
            summary = f"{previous_summary}\n{new_summary}"
        else:
            summary = previous_summary or new_summary

        if len(summary) > 2400:
            summary = summary[-2400:].lstrip()

        thread["messages"] = recent
        thread["summary"] = summary
        thread["compactions"] = int(thread["compactions"]) + 1
        thread["tokens"] = self._text_tokens(summary) + sum(
            self._message_tokens(item) for item in recent
        )

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id)
        if thread is None:
            return {"messages": [], "summary": "", "compactions": 0}

        return {
            "messages": list(thread["messages"]),
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        thread = self.state.get(thread_id)
        if thread is None:
            return 0
        return int(thread["compactions"])

    @staticmethod
    def _text_tokens(text: str) -> int:
        return estimate_tokens(text)

    @classmethod
    def _message_tokens(cls, message: dict[str, str]) -> int:
        return cls._text_tokens(message.get("role", "")) + cls._text_tokens(
            message.get("content", "")
        )
