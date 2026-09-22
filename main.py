"""하드코딩한 질문으로 기존 전체 평가 Graph를 실행한다."""

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.technical_research.node import research_technology
from kv_cache_eval.graph import build_graph


QUESTION = (
    "GPU 기반 클라우드 LLM 서비스 환경에서 KV cache 최적화 기술인 SW 진영의 KIVI와 "
    "HW 메모리 접근 진영의 InfiniGen을 논문과 공개 웹 자료로 조사해 줘. "
    "기술 원리·성능·한계의 근거와 출처를 확인하고, TRL·시장성·이해관계자·도메인 적합성을 "
    "비교한 보고서를 작성해 줘."
)


def main():
    graph = build_graph({
        "research_kivi": lambda state: {
            "kivi_evidence": research_technology(state, "KIVI", user_question=QUESTION)
        },
        "research_infinigen": lambda state: {
            "infinigen_evidence": research_technology(state, "InfiniGen", user_question=QUESTION)
        },
    })
    return graph.invoke(new_state())


if __name__ == "__main__":
    main()
