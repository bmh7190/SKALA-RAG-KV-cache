"""Shared substantive review criteria for report generation and judging."""

import re


def known_content_issues(report):
    """Flag narrow, observed misstatements; defer other semantics to the Judge."""
    issues = []
    for section, text in report["sections"]:
        sentences = re.split(r"(?<!\d)\.(?!\d)|\n", text)
        for sentence in sentences:
            if re.search(
                r"InfiniGen[^.]{0,280}(?:중요하지|덜 중요)[^.]{0,100}(?:버리|유지하지)",
                sentence,
            ):
                issues.append(
                    (
                        "groundedness",
                        section,
                        "선택적 prefetch와 영구 삭제를 혼동하는 문장",
                        "선택되지 않은 KV는 CPU에 유지되고 CPU pool 상한의 제거 정책은 별도라고 구분한다.",
                    )
                )
            if re.search(
                r"AI inference와 AI-optimized IaaS[^.]{0,100}학습과 추론", sentence
            ):
                issues.append(
                    (
                        "groundedness",
                        section,
                        "추론 시장과 학습 포함 IaaS의 범주를 혼동하는 문장",
                        "AI inference 시장은 추론 범주, AI-optimized IaaS는 학습과 추론 포함 범주라고 각각 설명한다.",
                    )
                )
    return list(dict.fromkeys(issues))


CONTENT_RULES = """내용 검토 규칙:
원문 감사 근거(source-audit)의 표·그림·실험 절을 확인한다. 기존 평가나 gaps와 충돌하면
원문을 우선하고 판단이 바뀐 이유를 설명한다. 검색에서 못 찾은 것은 논문에 없다는 뜻이 아니다.
InfiniGen의 batch size와 tokens per second 직접 결과를 원문에서 찾아 처리량에 반영한다.
원문에 직접 처리량이 있으면 speedup 환산 불가를 이유로 처리량 자체가 미확인이라고 쓰지 않는다.
메모리 수치는 모델 가중치 포함 여부, 배수는 baseline, 최대값은 조건과 범위를 명시한다.
최종 한계점에서도 KIVI의 peak memory와 throughput은 원문에서 확인된 항목이라고 명시한다.
KIVI의 메모리와 처리량을 전송량·latency와 함께 묶어서 미확인 또는 근거 부족이라고 쓰지 않는다.
각 성능 결과의 모델·GPU·CPU 메모리·연결·입출력 길이·배치를 가능한 범위에서 설명한다.
KIVI 초록의 2.6배 최대 메모리 요약과 p8 A100 배치·처리량 실험을 하나의 실험처럼 합치지 않는다.
InfiniGen p11의 1.63~32.93배는 여러 baseline에 대한 범위다. 전체 범위를 FlexGen 단독 대비라고 쓰지 않는다.
별도 원문 근거가 없으면 시퀀스 길이별 품질 실험의 모델·장비를 다른 성능 실험과 합치지 않는다.
InfiniGen의 prefetch에서 제외된 KV는 CPU에 남으며, CPU pool 상한 때의 제거는 별도 정책이라고 두 문장으로 명시한다.
기술 개요와 한계점에 이 두 구분 문장을 각각 넣는다. 생략하거나 모호한 버림 표현으로 대체하지 않는다.
금지 표현: 'InfiniGen은 중요하지 않은 entry를 버린다', '덜 중요한 항목은 유지하지 않는다'.
반드시 'GPU로 전송하지 않은 KV도 CPU pool에 남는다'고 명시한다.
AI inference market은 추론 시장이며 학습을 포함한다고 쓰지 않는다.
학습과 추론을 모두 포함하는 것은 AI-optimized IaaS다. 두 시장이 모두 학습을 포함한다고 쓰면 오류다.
시장 단위는 원문의 USD billion 또는 USD million을 유지한다. 억 달러로 환산하지 않는다.
조사 기준일과 확보한 자료 범위를 평가 방법에 적고, 검색·운영 검증의 미확인을 전 세계 사례 부재로 단정하지 않는다.
KIVI LongBench 평균은 모델명·설정·Table을 붙이고 과제별 하락도 설명한다.
평균이 유사하다고 모든 과제의 품질 유지나 통계적 유의성을 주장하지 않는다.
InfiniGen의 선택적 GPU 전송과 CPU pool 메모리 상한의 영구 제거 정책을 구분한다.
두 기술의 성숙도는 공개 구현·연구 실험·독립 검증·운영 검증이라는 같은 기준으로 비교한다.
근거 없이 한 기술을 더 초기 단계 또는 더 성숙하다고 단정하지 않는다.
이 보고서에서는 평가자가 부여한 숫자 TRL 및 운영 난이도 등급을 사용하지 않는다.
각 기술의 공개 구현·실험·독립 검증·운영 검증을 같은 기준으로 서술하고 차이가 미확인이면 그대로 표시한다.
사전 분석은 검색 범위를 알려주는 메타데이터다. 원문 감사로 확인된 결과가 사전 검색 미확인을 대체한다.
InfiniGen의 tokens/sec 값과 KIVI LongBench의 하락 사례는 실제 표·본문에서 숫자와 조건을 읽어 쓴다.
과제별 점수 차이가 큰데도 '약간씩 달라진다'고 축소하지 않는다.
InfiniGen의 GPU에 남는 partial query/key의 저장 부담과 CPU pool 정책도 원문에 따라 설명한다.
시장 수치는 기관별 범위·연도·단위·추정 및 전망임을 명시한다. AI-optimized IaaS는 학습과 추론을
포함하므로 추론 전용 시장과 동일시·합산하지 않는다. 시장 성장으로 기술 자체의 사업성을 단정하지 않는다.
채택 미확인은 확보한 검색 자료 범위로 한정한다. 수행하지 않은 검색·실험은 수행했다고 쓰지 않는다.
KIVI-sparsification 및 InfiniGen-FlexGen 호환성은 KIVI-InfiniGen 직접 결합의 증거가 아니다.
추천·우선 도입·우열 판정 대신 분석 대상 병목과 조건을 설명한다.
관측 사실, 저자 주장, 분석 추론, 조사 미확인을 구분하고 기대 편익은 실제 운영 결과처럼 쓰지 않는다.
성숙도·시장성·이해관계자·도메인 모두 두 기술을 다룬다. 이해관계자 다섯 집단 각각의 편익과 부담을
구분하고, 도메인 여섯 항목(GPU 메모리 사용량, 데이터 전송량, 추론 지연시간, 처리량, 모델 품질,
적용·운영 난이도) 각각의 결과·실험 조건·한계를 설명한다.
한계점에는 원문 미확인/이번 검색 미확인/운영 검증 필요를 구분하고 재현 가능한 향후 실험안을 적는다.
Markdown # 기호와 표를 쓰지 않는다. 참고문헌은 코드가 생성한다."""
