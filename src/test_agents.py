from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    config = load_config(tmp_path)
    config.state_dir = tmp_path / "state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 80
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    #TODO: verify `User.md` can be created, updated, and edited.

    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")
    user_id = "dung"

    # Nothing persisted yet: no file, default content.
    assert store.file_size(user_id) == 0
    assert store.read_text(user_id).startswith("# User Profile")

    # Create the markdown file by writing a fact.
    store.upsert_fact(user_id, "name", "Dũng")
    assert store.path_for(user_id).exists()
    assert store.file_size(user_id) > 0
    assert store.facts(user_id)["name"] == "Dũng"

    # Update an existing fact in place (no duplicate key).
    store.upsert_fact(user_id, "name", "Dũng CT")
    store.upsert_fact(user_id, "location", "Hà Nội")
    facts = store.facts(user_id)
    assert facts["name"] == "Dũng CT"
    assert facts["location"] == "Hà Nội"
    assert sum(1 for line in store.read_text(user_id).splitlines() if "name:" in line) == 1

    # Edit raw markdown content directly.
    assert store.edit_text(user_id, "Hà Nội", "Đà Nẵng") is True
    assert store.facts(user_id)["location"] == "Đà Nẵng"
    # Editing a missing snippet is a no-op.
    assert store.edit_text(user_id, "không tồn tại", "x") is False


def test_compact_trigger(tmp_path: Path) -> None:
    #TODO: verify long threads trigger compaction.

    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    thread_id = "thread-compact"

    # A brand-new thread has not compacted anything yet.
    assert agent.compaction_count(thread_id) == 0

    # Feed enough long messages to cross the (small) token threshold.
    for index in range(12):
        agent.reply("dung", thread_id, f"Tin số {index}: " + "nội dung nền dài dòng " * 6)

    # Older messages must have been folded into the compact summary.
    assert agent.compaction_count(thread_id) > 0
    context = agent.compact_memory.context(thread_id)
    recent_messages = context["messages"]
    assert isinstance(recent_messages, list)
    assert context["summary"]
    assert len(recent_messages) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    #TODO: verify advanced remembers across sessions and baseline does not.

    config = make_config(tmp_path)
    user_id = "dung"

    # Advanced: state the name in one session/thread...
    advanced_session_one = AdvancedAgent(config=config, force_offline=True)
    advanced_session_one.reply(user_id, "thread-a", "Mình tên là Dũng")

    # ...a fresh agent instance (new session) reading the same state_dir recalls it.
    advanced_session_two = AdvancedAgent(config=config, force_offline=True)
    recalled = advanced_session_two.reply(user_id, "thread-b", "Bạn còn nhớ tên mình không?")
    assert "Dũng" in recalled["response"]

    # Baseline only knows the current thread, so a new thread forgets the name.
    baseline = BaselineAgent(config=config, force_offline=True)
    baseline.reply(user_id, "thread-a", "Mình tên là Dũng")
    forgotten = baseline.reply(user_id, "thread-b", "Bạn còn nhớ tên mình không?")
    assert "Dũng" not in forgotten["response"]
    assert "chưa thấy" in forgotten["response"].casefold()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    #TODO: compare prompt load of baseline vs advanced on a long thread.

    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    thread_id = "thread-long"

    messages = [f"Tin tức số {index}: " + "thông tin nền dài dòng " * 6 for index in range(20)]
    for message in messages:
        baseline.reply("dung", thread_id, message)
        advanced.reply("dung", thread_id, message)

    # Advanced compacts; baseline keeps the full transcript in every prompt.
    assert advanced.compaction_count(thread_id) > 0
    assert baseline.compaction_count(thread_id) == 0

    # Compaction keeps the cumulative prompt load below the baseline.
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
