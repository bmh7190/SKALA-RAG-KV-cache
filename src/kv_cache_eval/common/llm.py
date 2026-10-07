"""Shared LangChain model factory; imports and credentials are resolved at call time."""

import os

from kv_cache_eval.common.config import load_environment


def chat_model(*, role: str = "generator", timeout: float = 120, settings=None):
    from langchain_openai import ChatOpenAI

    if settings is not None:
        return ChatOpenAI(**settings)
    load_environment()
    if os.getenv("LLM_PROVIDER", "").strip().lower() != "openai":
        raise ValueError("LLM_PROVIDER=openai 설정이 필요합니다")
    model = os.getenv("JUDGE_MODEL" if role == "judge" else "LLM_MODEL") or os.getenv(
        "LLM_MODEL"
    )
    if not model or not os.getenv("OPENAI_API_KEY"):
        raise ValueError("LLM_MODEL과 OPENAI_API_KEY 설정이 필요합니다")
    # Graph-level retries remain visible in checkpoints and traces.
    return ChatOpenAI(model=model.strip(), timeout=timeout, max_retries=0)


def structured_chain(
    schema, system: str, *, role="generator", name="structured_generation"
):
    """Use native prompt templates and structured output; preserve LangChain tracing."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{payload}")]
    )
    return (
        prompt
        | chat_model(role=role).with_structured_output(schema, method="json_schema")
    ).with_config(run_name=name)
