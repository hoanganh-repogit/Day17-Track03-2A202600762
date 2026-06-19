from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-3.1-flash-lite",
    "anthropic": "claude-3-haiku-20240307",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(root / ".env")

    def make_provider_config(prefix: str = "", fallback: ProviderConfig | None = None) -> ProviderConfig:
        provider = (
            os.getenv(f"{prefix}LLM_PROVIDER")
            or os.getenv(f"{prefix}PROVIDER")
            or (fallback.provider if fallback else "openai")
        )
        provider = provider.strip().lower()
        if provider == "anthorpic":
            provider = "anthropic"
        if provider not in DEFAULT_MODELS:
            supported = ", ".join(DEFAULT_MODELS)
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported}")

        model_name = (
            os.getenv(f"{prefix}LLM_MODEL")
            or os.getenv(f"{prefix}MODEL")
            or (fallback.model_name if fallback and fallback.provider == provider else DEFAULT_MODELS[provider])
        )
        temperature = float(
            os.getenv(f"{prefix}LLM_TEMPERATURE")
            or os.getenv("LLM_TEMPERATURE")
            or (str(fallback.temperature) if fallback else "0")
        )
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "custom": os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "ollama": None,
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
        }
        base_urls = {
            "custom": os.getenv("CUSTOM_BASE_URL"),
            "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "openrouter": os.getenv("OPENROUTER_BASE_URL"),
        }
        return ProviderConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_keys[provider],
            base_url=base_urls.get(provider),
        )

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    model = make_provider_config()

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        model=model,
        judge_model=make_provider_config("JUDGE_", model),
    )
