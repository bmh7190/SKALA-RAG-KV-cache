"""Shared LangChain model factory; imports and credentials are resolved at call time."""
import os

from kv_cache_eval.common.config import load_environment


def chat_model(*, role: str = 'generator', timeout: float = 120):
    from langchain_openai import ChatOpenAI

    load_environment()
    if os.getenv('LLM_PROVIDER', '').strip().lower() != 'openai':
        raise ValueError('LLM_PROVIDER=openai 설정이 필요합니다')
    model = os.getenv('JUDGE_MODEL' if role == 'judge' else 'LLM_MODEL') or os.getenv('LLM_MODEL')
    if not model or not os.getenv('OPENAI_API_KEY'):
        raise ValueError('LLM_MODEL과 OPENAI_API_KEY 설정이 필요합니다')
    # Graph-level retries remain visible in checkpoints and traces.
    return ChatOpenAI(model=model.strip(), timeout=timeout, max_retries=0)
