from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """
    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """
        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = self._estimate_prompt_context_tokens(thread_id)

        result = self.langchain_agent.invoke(
            {"messages": session.messages},
            config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
        )
        response = self._extract_response_text(result)
        session.messages.append({"role": "assistant", "content": response})

        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": response,
            "answer": response,
            "content": response,
            "thread_id": thread_id,
            "agent_tokens": response_tokens,
            "token_usage": session.token_usage,
            "total_agent_tokens": session.token_usage,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": session.prompt_tokens_processed,
            "total_prompt_tokens": session.prompt_tokens_processed,
        }

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """
        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = self._estimate_prompt_context_tokens(thread_id)

        response = self._offline_response(thread_id, message)
        session.messages.append({"role": "assistant", "content": response})

        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": response,
            "answer": response,
            "content": response,
            "thread_id": thread_id,
            "agent_tokens": response_tokens,
            "token_usage": session.token_usage,
            "total_agent_tokens": session.token_usage,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_processed": session.prompt_tokens_processed,
            "total_prompt_tokens": session.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        """
        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        try:
            model = build_chat_model(self.config.model)
        except Exception:
            return None

        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent
        except Exception:
            return None

        try:
            return create_react_agent(model, tools=[], checkpointer=InMemorySaver())
        except Exception:
            return None

    def _estimate_prompt_context_tokens(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        if session is None:
            return 0
        return sum(estimate_tokens(message["content"]) for message in session.messages)

    def _offline_response(self, thread_id: str, message: str) -> str:
        lowered = message.casefold()
        session = self.sessions[thread_id]
        user_messages = [item["content"] for item in session.messages if item["role"] == "user"]
        current_thread_text = "\n".join(user_messages)

        if any(term in lowered for term in ("tên", "name")):
            name = self._latest_fact(
                current_thread_text,
                (
                    "mình tên là ",
                    "tôi tên là ",
                    "tớ tên là ",
                    "my name is ",
                    "tên mình là ",
                    "tên của mình là ",
                ),
            )
            if name:
                return f"Trong thread này, mình thấy bạn tên là {name}."
            return "Mình chưa thấy tên của bạn trong thread này."

        if any(term in lowered for term in ("ở đâu", "nơi ở", "location", "đang ở")):
            location = self._latest_fact(
                current_thread_text,
                (
                    "mình đang ở ",
                    "mình ở ",
                    "tôi đang ở ",
                    "tôi ở ",
                    "hiện ở ",
                    "currently in ",
                    "i live in ",
                ),
            )
            if location:
                return f"Trong thread này, mình thấy bạn đang ở {location}."
            return "Mình chưa thấy nơi ở của bạn trong thread này."

        if any(term in lowered for term in ("nghề", "làm nghề", "công việc", "job", "profession")):
            profession = self._latest_fact(
                current_thread_text,
                (
                    "mình đang làm ",
                    "mình làm ",
                    "tôi đang làm ",
                    "tôi làm ",
                    "nghề nghiệp là ",
                    "my job is ",
                    "i work as ",
                ),
            )
            if profession:
                return f"Trong thread này, mình thấy nghề hiện tại của bạn là {profession}."
            return "Mình chưa thấy nghề nghiệp của bạn trong thread này."

        if any(term in lowered for term in ("style", "phong cách", "kiểu trả lời", "trả lời")):
            style = self._latest_style(current_thread_text)
            if style:
                return f"Trong thread này, mình thấy bạn thích mình trả lời {style}."
            return "Mình sẽ trả lời ngắn gọn và rõ ý trong thread này."

        return "Mình đã ghi nhận trong thread hiện tại. Sang thread mới thì baseline sẽ không giữ thông tin này."

    @staticmethod
    def _latest_fact(text: str, prefixes: tuple[str, ...]) -> str | None:
        latest: str | None = None
        folded = text.casefold()
        for prefix in prefixes:
            start = 0
            folded_prefix = prefix.casefold()
            while True:
                index = folded.find(folded_prefix, start)
                if index == -1:
                    break
                value_start = index + len(prefix)
                value_end = len(text)
                for delimiter in ".!?\n;,":
                    delimiter_index = text.find(delimiter, value_start)
                    if delimiter_index != -1:
                        value_end = min(value_end, delimiter_index)
                value = text[value_start:value_end].strip()
                if value:
                    latest = value
                start = value_start
        return latest

    @staticmethod
    def _latest_style(text: str) -> str | None:
        candidates = (
            "ngắn gọn",
            "rõ ý",
            "có ví dụ thực tế",
            "có ví dụ thực chiến",
            "bullet",
            "3 bullet",
        )
        found = [candidate for candidate in candidates if candidate.casefold() in text.casefold()]
        return ", ".join(found) if found else None

    @staticmethod
    def _extract_response_text(result: Any) -> str:
        if isinstance(result, dict) and result.get("messages"):
            last_message = result["messages"][-1]
            content = getattr(last_message, "content", None)
            if content is not None:
                return str(content)
            if isinstance(last_message, dict):
                return str(last_message.get("content", ""))

        content = getattr(result, "content", None)
        if content is not None:
            return str(content)

        return str(result)
