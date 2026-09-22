"""도메인 적용 평가 기준. 팀의 「KV cache 최적화 기술 평가 설계서」 4~6쪽."""

from decimal import Context, Decimal, InvalidOperation, localcontext
from fractions import Fraction

DOMAIN_RUBRIC = {
    "GPU 메모리 사용량": {
        "question": "KV Cache로 인한 GPU 메모리 부담을 얼마나 줄일 수 있는가?",
        "scores": {
            5: "GPU 메모리 사용량 50% 이상 감소",
            4: "30~50% 미만 감소",
            3: "10~30% 미만 감소",
            2: "0~10% 미만 감소",
            1: "감소 효과가 없거나 오히려 증가",
        },
        "evidence": "메모리 사용량, Peak Memory, KV Cache Size",
    },
    "데이터 전송량": {
        "question": "CPU↔GPU 간 KV Cache 데이터 이동량을 얼마나 줄일 수 있는가?",
        "scores": {
            5: "데이터 전송량 50% 이상 감소",
            4: "30~50% 미만 감소",
            3: "10~30% 미만 감소",
            2: "0~10% 미만 감소",
            1: "감소 효과가 없거나 데이터 이동량 증가",
        },
        "evidence": "Host↔GPU Transfer, PCIe Traffic, Prefetch 데이터량",
    },
    "추론 지연시간": {
        "question": "기술 적용 후 응답 생성 시간이 얼마나 변화하는가?",
        "scores": {
            5: "지연시간 30% 이상 감소",
            4: "10~30% 미만 감소",
            3: "기존과 유사한 수준 (±10%)",
            2: "지연시간 10~30% 증가",
            1: "지연시간 30% 이상 증가",
        },
        "evidence": "Latency, Token Generation Time, End-to-End Inference Time",
    },
    "처리량": {
        "question": "동시에 여러 요청을 처리하는 성능이 얼마나 변화하는가?",
        "scores": {
            5: "처리량 50% 이상 증가",
            4: "30~50% 미만 증가",
            3: "10~30% 미만 증가",
            2: "변화가 거의 없음 (±10%)",
            1: "처리량 감소",
        },
        "evidence": "Throughput, Tokens/sec, Requests/sec, Batch Size 증가 효과",
    },
    "모델 품질": {
        "question": "적용 후에도 기존 모델의 정확도나 생성 품질이 유지되는가?",
        "scores": {
            5: "기존 모델과 거의 동일한 품질 유지",
            4: "품질 저하가 매우 작아 실제 사용에 영향이 거의 없음",
            3: "일부 Task에서 소폭 품질 저하 발생",
            2: "여러 Task에서 명확한 품질 저하 발생",
            1: "정확도·생성 품질 저하가 커 실제 적용이 어려움",
        },
        "evidence": "Accuracy, Perplexity, Benchmark Score, 생성 품질 평가",
    },
    "적용·운영 난이도": {
        "question": "기존 LLM 서비스에 적용하기 위한 변경과 추가 관리 부담은 어느 정도인가?",
        "scores": {
            5: "기존 시스템에 거의 수정 없이 적용 가능",
            4: "일부 라이브러리 또는 설정 변경만 필요",
            3: "모델·추론 코드 일부 수정 및 추가 설정 필요",
            2: "시스템 구조 변경 또는 별도 메모리 관리 필요",
            1: "대규모 구조 변경, 특수 환경·하드웨어 등이 필요",
        },
        "evidence": "설치 과정, 코드 변경량, 추가 하드웨어 요구, 호환성, 운영 관리 요소",
    },
}

NUMERIC_CRITERIA = tuple(DOMAIN_RUBRIC)[:4]


def calculate_score(criterion, baseline, optimized):
    """비교 가능한 측정값으로 계산한다. 원문 구간이 겹치면 점수를 확정하지 않는다."""
    if criterion not in NUMERIC_CRITERIA:
        raise ValueError("정량 평가 항목이 아닙니다")
    if isinstance(baseline, bool) or isinstance(optimized, bool):
        raise ValueError("측정값은 숫자여야 합니다")
    try:
        before, after = Decimal(str(baseline)), Decimal(str(optimized))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("측정값은 유한한 숫자여야 합니다") from exc
    if not before.is_finite() or not after.is_finite() or before <= 0 or after < 0:
        raise ValueError("기준값은 양수, 적용값은 0 이상의 유한한 숫자여야 합니다")
    before_ratio, after_ratio = Fraction(before), Fraction(after)
    change = (after_ratio - before_ratio) / before_ratio * 100
    if criterion != "처리량":
        change = -change
    # 점수는 정확한 분수로 결정하고 표시용 소수만 별도로 계산한다.
    # 외부 코드가 바꾼 Decimal 정밀도·예외 설정을 상속하지 않는다.
    precision = max(28, len(str(abs(change.numerator))) + len(str(change.denominator)) + 8)
    with localcontext(Context(prec=precision)):
        displayed = Decimal(change.numerator) / Decimal(change.denominator)
    return _score_change(criterion, change), displayed


def score_change(criterion, change):
    """처리량은 증가율, 나머지 항목은 감소율을 받아 원문 기준으로 채점한다."""
    if criterion not in NUMERIC_CRITERIA or isinstance(change, bool):
        raise ValueError("정량 항목과 유한한 변화율이 필요합니다")
    try:
        change = Decimal(str(change))
    except InvalidOperation as exc:
        raise ValueError("변화율은 숫자여야 합니다") from exc
    if not change.is_finite():
        raise ValueError("변화율은 유한해야 합니다")
    return _score_change(criterion, change)


def _score_change(criterion, change):
    if criterion in ("GPU 메모리 사용량", "데이터 전송량"):
        matches = (change >= 50, 30 <= change < 50, 10 <= change < 30, 0 <= change < 10, change <= 0)
    elif criterion == "추론 지연시간":
        matches = (change >= 30, 10 <= change < 30, -10 <= change <= 10, -30 <= change <= -10, change <= -30)
    else:
        matches = (change >= 50, 30 <= change < 50, 10 <= change < 30, -10 <= change <= 10, change < 0)
    scores = [score for score, matched in zip((5, 4, 3, 2, 1), matches) if matched]
    return scores[0] if len(scores) == 1 else None
