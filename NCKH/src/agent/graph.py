from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable

from llama_index.core import Settings
from llama_index.llms.groq import Groq

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from agent.nodes import get_or_build_index
    from agent.state import AgentConfig
else:
    from .nodes import get_or_build_index
    from .state import AgentConfig


class NCKHAgent:
    def __init__(self, config: AgentConfig, rebuild_index: bool = False) -> None:
        self.config = config
        self.index = get_or_build_index(config, rebuild=rebuild_index)
        self.retriever = self.index.as_retriever(similarity_top_k=config.similarity_top_k)
        self.llm_pool = self._build_llm_pool()
        self.llm = self.llm_pool[0] if self.llm_pool else Settings.llm
        self.history: list[tuple[str, str]] = []

    def _build_llm_pool(self) -> list[object]:
        candidates = [
            self.config.groq_model,
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ]
        llms: list[object] = []
        for model_name in dict.fromkeys(candidates):
            try:
                llms.append(
                    Groq(
                        model=model_name,
                        api_key=self.config.groq_api_key,
                        temperature=self.config.temperature,
                    )
                )
            except Exception:
                continue
        return llms

    def reset(self) -> None:
        self.history.clear()

    def _history_text(self) -> str:
        if not self.history or self.config.max_history_turns <= 0:
            return "(no previous conversation)"
        turns = self.history[-self.config.max_history_turns :]
        lines: list[str] = []
        for i, (user_q, assistant_a) in enumerate(turns, start=1):
            lines.append(f"Turn {i} - User: {user_q}")
            lines.append(f"Turn {i} - Assistant: {assistant_a}")
        return "\n".join(lines)

    def _to_text(self, response_obj: object) -> str:
        text = getattr(response_obj, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(response_obj).strip()

    def _rewrite_question(self, question: str) -> str:
        if not self.config.rewrite_query:
            return question
        if not self.history:
            return question
        prompt = (
            "Viet lai cau hoi thanh truy van tim kiem doc lap bang tieng Viet.\n"
            "Giu nguyen y dinh va cac thuc the quan trong tu lich su hoi thoai.\n"
            "Chi tra ve 1 dong truy van, khong giai thich.\n\n"
            f"Chat history:\n{self._history_text()}\n\n"
            f"User question:\n{question}\n"
        )
        try:
            rewritten = self._to_text(self._complete(prompt))
            return rewritten if rewritten else question
        except Exception:
            return question

    def _is_quota_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "resource_exhausted" in msg
            or "quota" in msg
            or "429" in msg
            or "rate limit" in msg
        )

    def _is_transient_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "timeout" in msg
            or "temporar" in msg
            or "connection" in msg
            or "unavailable" in msg
            or "500" in msg
            or "502" in msg
            or "503" in msg
            or "504" in msg
        )

    def _complete(self, prompt: str) -> object:
        if not self.llm_pool:
            return self.llm.complete(prompt)
        last_exc: Exception | None = None
        for llm in self.llm_pool:
            for attempt in range(2):
                try:
                    self.llm = llm
                    return llm.complete(prompt)
                except Exception as exc:
                    last_exc = exc
                    if self._is_quota_error(exc):
                        break
                    if self._is_transient_error(exc) and attempt == 0:
                        time.sleep(0.6)
                        continue
                    continue
        if last_exc is not None:
            raise last_exc
        return self.llm.complete(prompt)

    def _build_context(self, retrieved_nodes: Iterable[object]) -> tuple[str, list[str]]:
        chunks: list[str] = []
        sources: list[str] = []
        used = 0
        for idx, node_ws in enumerate(retrieved_nodes, start=1):
            node = getattr(node_ws, "node", node_ws)
            text = ""
            if hasattr(node, "get_content"):
                text = node.get_content().strip()
            if not text:
                text = str(node).strip()
            if not text:
                continue

            metadata = getattr(node, "metadata", {}) or {}
            file_source = metadata.get("file_source", "unknown_source")
            source_label = f"[S{idx}] {file_source}"
            snippet = f"{source_label}\n{text}"
            if used + len(snippet) > self.config.max_context_chars:
                break
            chunks.append(snippet)
            sources.append(source_label)
            used += len(snippet)

        if not chunks:
            return "Khong tim thay ngu canh phu hop trong du lieu.", []
        return "\n\n".join(chunks), sources

    def ask(self, question: str, include_sources: bool = True) -> str:
        question = question.strip()
        if not question:
            return "Ban hay nhap cau hoi cu the de minh ho tro."

        standalone_question = self._rewrite_question(question)
        try:
            retrieved_nodes = self.retriever.retrieve(standalone_question)
        except Exception:
            retrieved_nodes = []
        context_text, sources = self._build_context(retrieved_nodes)

        prompt = (
            "Ban la tro ly tu van hoc vu dai hoc danh cho sinh vien.\n"
            "Tra loi bang tieng Viet tu nhien, ro rang, de hieu.\n"
            "Uu tien cach viet ngan gon, co cau truc, de sinh vien de ap dung.\n"
            "Neu co so lieu, nen trinh bay theo bullet ngan.\n"
            "Quy tac:\n"
            "- Chi dung thong tin co trong context duoc cung cap.\n"
            "- Neu thieu du lieu, noi ro dang thieu gi va goi y sinh vien can cung cap them gi.\n"
            "- Khong doan mo them thong tin ngoai context.\n\n"
            f"Conversation history:\n{self._history_text()}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            f"Current user question:\n{question}\n\n"
            "Final answer:"
        )

        try:
            answer = self._to_text(self._complete(prompt))
        except Exception:
            answer = (
                "He thong dang ban tam thoi khi goi mo hinh. "
                "Ban vui long thu lai sau 5-10 giay."
            )

        if self.config.max_history_turns > 0:
            self.history.append((question, answer))
            if len(self.history) > self.config.max_history_turns:
                self.history = self.history[-self.config.max_history_turns :]

        if include_sources and sources:
            source_lines = "\n".join(sources[: min(self.config.max_sources_in_answer, len(sources))])
            return f"{answer}\n\nNguon tham khao:\n{source_lines}"
        return answer
