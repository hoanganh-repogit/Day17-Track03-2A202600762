from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower()
    return {"anthorpic": "anthropic"}.get(provider, provider)


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)
    common_kwargs = {
        "temperature": config.temperature,
    }
    if config.api_key:
        common_kwargs["api_key"] = config.api_key

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model_name, **common_kwargs)
    if provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model_name, base_url=config.base_url, **common_kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.model_name, **common_kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model_name=config.model_name, **common_kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.model_name, base_url=config.base_url, temperature=config.temperature)
    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(model=config.model_name, base_url=config.base_url, **common_kwargs)

    supported = "openai, custom, gemini, anthropic, ollama, openrouter"
    raise ValueError(f"Unsupported provider '{config.provider}'. Supported providers: {supported}")
