from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_bytes())


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0

    answer_normalized = answer.casefold()
    matched = sum(1 for fact in expected if fact.casefold() in answer_normalized)

    if matched == 0:
        return 0.0
    if matched == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    answer = answer.strip()
    if not answer:
        return 0.0

    expected_facts = [fact.strip().casefold() for fact in expected if fact.strip()]
    if not expected_facts:
        return 1.0

    answer_normalized = answer.casefold()
    matched = sum(1 for fact in expected_facts if fact in answer_normalized)
    score = matched / len(expected_facts)

    word_count = len(answer.split())
    if word_count < len(expected_facts):
        score *= 0.7
    elif word_count > 120:
        score *= 0.85

    uncertain_markers = ("i don't know", "not sure", "cannot answer", "khong biet")
    if any(marker in answer_normalized for marker in uncertain_markers):
        score *= 0.8

    return round(max(0.0, min(1.0, score)), 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    def memory_size(user_id: str) -> int:
        size_fn = getattr(agent, "memory_file_size", None)
        return int(size_fn(user_id)) if size_fn is not None else 0

    def metric(result: dict[str, Any], key: str) -> int:
        return int(result.get(key) or 0)

    user_ids = {str(item.get("user_id") or "benchmark_user") for item in conversations}

    # Start each run from a clean persistent profile for the benchmark users so the
    # measured memory growth reflects only what THIS run writes. Without this, repeated
    # runs reuse the persisted files (idempotent upserts) and report growth ~= 0.
    profile_store = getattr(agent, "profile_store", None)
    if profile_store is not None:
        for user_id in user_ids:
            profile_store.path_for(user_id).unlink(missing_ok=True)

    memory_before = sum(memory_size(user_id) for user_id in user_ids)

    reply = agent.reply
    agent_tokens_only = 0
    prompt_tokens_processed = 0
    recall_total = 0.0
    quality_total = 0.0
    recall_count = 0
    thread_ids: list[str] = []
    prefix = f"benchmark-{agent_name.casefold().replace(' ', '-')}-{id(agent)}"

    for index, conversation in enumerate(conversations):
        user_id = str(conversation.get("user_id") or "benchmark_user")
        conversation_id = str(conversation.get("id") or index)
        thread_id = f"{prefix}-{conversation_id}"
        recall_thread_id = f"{thread_id}-recall"
        thread_ids.extend((thread_id, recall_thread_id))

        for turn in conversation.get("turns", []):
            result = reply(user_id, thread_id, str(turn))
            agent_tokens_only += metric(result, "agent_tokens")
            prompt_tokens_processed += metric(result, "prompt_tokens")

        for item in conversation.get("recall_questions", []):
            question = str(item.get("question") or "")
            expected = [str(fact) for fact in item.get("expected_contains", [])]
            result = reply(user_id, recall_thread_id, question)
            answer = str(result.get("answer") or result.get("response") or result.get("content") or "")

            agent_tokens_only += metric(result, "agent_tokens")
            prompt_tokens_processed += metric(result, "prompt_tokens")
            recall_total += recall_points(answer, expected)
            quality_total += heuristic_quality(answer, expected)
            recall_count += 1

    compaction_fn = getattr(agent, "compaction_count", None)
    compactions = (
        sum(int(compaction_fn(thread_id)) for thread_id in thread_ids)
        if compaction_fn is not None
        else 0
    )
    memory_after = sum(memory_size(user_id) for user_id in user_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens_only,
        prompt_tokens_processed=prompt_tokens_processed,
        recall_score=round(recall_total / recall_count, 3) if recall_count else 0.0,
        response_quality=round(quality_total / recall_count, 3) if recall_count else 0.0,
        memory_growth_bytes=max(0, memory_after - memory_before),
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    lines = [
        "| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | "
        f"{row.recall_score:.3f} | {row.response_quality:.3f} | "
        f"{row.memory_growth_bytes} | {row.compactions} |"
        for row in rows
    )
    return "\n".join(lines)


def _force_offline() -> bool:
    """Use the deterministic path unless BENCHMARK_USE_LLM is truthy.

    Set BENCHMARK_USE_LLM=1 (and configure provider/API key in .env) to run the
    benchmark against a real LLM. Defaults to offline so runs stay reproducible.
    """

    return os.getenv("BENCHMARK_USE_LLM", "").strip().lower() not in {"1", "true", "yes", "on"}


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    force_offline = _force_offline()
    mode = "offline (deterministic)" if force_offline else f"live LLM ({config.model.provider}:{config.model.model_name})"
    suites = (
        ("Standard benchmark", config.data_dir / "conversations.json"),
        ("Long-context stress benchmark", config.data_dir / "advanced_long_context.json"),
    )

    print(f"Benchmark mode: {mode}")

    for title, dataset_path in suites:
        conversations = load_conversations(dataset_path)
        rows = [
            run_agent_benchmark(
                "Baseline",
                BaselineAgent(config=config, force_offline=force_offline),
                conversations,
                config,
            ),
            run_agent_benchmark(
                "Advanced",
                AdvancedAgent(config=config, force_offline=force_offline),
                conversations,
                config,
            ),
        ]

        print(f"\n## {title}")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
