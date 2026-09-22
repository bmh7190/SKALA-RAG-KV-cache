"""도메인 적용 평가에 사용하는 6가지 기준.

RAG-Design 설계서(판교 3조) 3.2절 "도메인 적용 관점"의 평가 기준·점수 척도를
그대로 반영한다. 선정 도메인은 domain_and_criteria["domain"]으로 State에 이미
들어있다 (기본값: "GPU 기반 클라우드 LLM 서비스").
"""

from typing import NamedTuple


class DomainCriterion(NamedTuple):
    name: str
    question: str
    rubric: str  # 5~1점 기준 요약
    evidence_hint: str  # 어떤 근거를 확인해야 하는지


DOMAIN_CRITERIA: tuple[DomainCriterion, ...] = (
    DomainCriterion(
        name="GPU 메모리 사용량",
        question="KV Cache로 인한 GPU 메모리 부담을 얼마나 줄일 수 있는가?",
        rubric=(
            "5점: 50% 이상 감소 / 4점: 30~50% 미만 감소 / 3점: 10~30% 미만 감소 / "
            "2점: 0~10% 미만 감소 / 1점: 감소 효과 없거나 오히려 증가"
        ),
        evidence_hint="논문의 메모리 사용량, Peak Memory, KV Cache Size 등의 실험 결과",
    ),
    DomainCriterion(
        name="데이터 전송량",
        question="CPU↔GPU 간 KV Cache 데이터 이동량을 얼마나 줄일 수 있는가?",
        rubric=(
            "5점: 50% 이상 감소 / 4점: 30~50% 미만 감소 / 3점: 10~30% 미만 감소 / "
            "2점: 0~10% 미만 감소 / 1점: 감소 효과 없거나 데이터 이동량 증가"
        ),
        evidence_hint="Host↔GPU Transfer, PCIe Traffic, Prefetch 데이터량 등의 결과",
    ),
    DomainCriterion(
        name="추론 지연시간",
        question="기술 적용 후 응답 생성 시간이 얼마나 변화하는가?",
        rubric=(
            "5점: 지연시간 30% 이상 감소 / 4점: 10~30% 미만 감소 / 3점: 기존과 유사한 수준(±10%) / "
            "2점: 지연시간 10~30% 증가 / 1점: 지연시간 30% 이상 증가"
        ),
        evidence_hint="Latency, Token Generation Time, End-to-End Inference Time",
    ),
    DomainCriterion(
        name="처리량",
        question="동시에 여러 요청을 처리하는 성능이 얼마나 변화하는가?",
        rubric=(
            "5점: 처리량 50% 이상 증가 / 4점: 30~50% 미만 증가 / 3점: 10~30% 미만 증가 / "
            "2점: 변화가 거의 없음(±10%) / 1점: 처리량 감소"
        ),
        evidence_hint="Throughput, Tokens/sec, Requests/sec, Batch Size 증가 효과",
    ),
    DomainCriterion(
        name="모델 품질",
        question="최적화 기술 적용 후에도 기존 모델의 정확도나 생성 품질이 유지되는가?",
        rubric=(
            "5점: 기존 모델과 거의 동일한 품질 유지 / 4점: 품질 저하가 매우 작아 실제 사용에 영향 거의 없음 / "
            "3점: 일부 Task에서 소폭 품질 저하 / 2점: 여러 Task에서 명확한 품질 저하 / "
            "1점: 정확도·생성 품질 저하가 커서 실제 적용이 어려움"
        ),
        evidence_hint="Accuracy, Perplexity, Benchmark Score, 생성 품질 평가",
    ),
    DomainCriterion(
        name="적용·운영 난이도",
        question="기존 LLM 서비스에 적용하기 위해 얼마나 많은 변경과 추가 관리가 필요한가?",
        rubric=(
            "5점: 기존 시스템에 거의 수정 없이 적용 가능 / 4점: 일부 라이브러리 또는 설정 변경만 필요 / "
            "3점: 모델·추론 코드 일부 수정 및 추가 설정 필요 / 2점: 시스템 구조 변경 또는 별도 메모리 관리 필요 / "
            "1점: 대규모 구조 변경, 특수 환경·하드웨어 등이 필요"
        ),
        evidence_hint="설치 과정, 코드 변경량, 추가 하드웨어 요구, 호환성, 운영 관리 요소",
    ),
)
