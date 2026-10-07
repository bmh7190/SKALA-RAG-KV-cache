"""Shared LangChain model factory; imports and credentials are resolved at call time."""

from kv_cache_eval.common.config import load_environment, model_settings


def chat_model(*, role: str = "generator", timeout: float = 120, settings=None):
    from langchain_openai import ChatOpenAI

    if settings is not None:
        return ChatOpenAI(**settings)
    load_environment()
    # Graph-level retries remain visible in checkpoints and traces.
    return ChatOpenAI(**model_settings(role=role, timeout=timeout))


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
