# 도메인 적용 평가

GPU 기반 클라우드 LLM 서비스에 KIVI와 InfiniGen을 적용할 때의 메모리, 전송량, 지연시간, 처리량, 모델 품질, 운영 난이도를 평가합니다. 평가 기준과 점수 구간은 `rubric.py`가 정의합니다.

## 코드 구조와 읽는 순서

`node.py`의 `evaluate()`부터 읽습니다. 해당 함수는 평가할 범위의 결과를 얻고, 부분 재평가라면 이전 결과와 병합합니다. `_evaluate_scope()`에서 근거 준비 → 요청별 평가 → 미복구 오류 확인 → State 갱신값 조립 순서를 확인할 수 있습니다.

| 파일 | 책임 |
|---|---|
| `node.py` | State 입력과 평가 단계 연결, 부분 결과 병합 |
| `evidence.py` | 근거 출처·형식·중복 검사, `CollectedEvidence`로 수집 결과 반환 |
| `assessment.py` | 입력 예산에 맞춘 요청 구성, 모델 초안·검토·항목별 한 번 보정 |
| `validation.py` | 초안 인용·측정값 검증, 검토 결과에 따른 판단 확정 |
| `results.py` | 미확인 평가와 최종 `domain_eval`·`domain_evidence` 조립 |
| `measurement.py`, `rubric.py` | 수치 계산·측정값 조건과 점수 기준 |
| `prompt.py`, `runtime.py` | 모델 출력 계약과 설정·실제 호출 |

`assessment.py`는 평가 행과 진단 목록을 갱신합니다. `evidence.py`는 입력 근거를 복사해 검사하고, 노드가 받은 State 자체는 변경하지 않습니다. 외부 호출은 `runtime.invoke_structured()`를 통합니다. 검증 테스트는 호출을 사용하는 `assessment.invoke_structured`를 대체합니다.

## State 연결

`node.evaluate(state)`는 `domain_and_criteria["domain"]`, `kivi_evidence`, `infinigen_evidence`를 읽고 `domain_evidence`와 `domain_eval`만 반환합니다. `market_evidence`나 `market_eval`은 읽거나 수정하지 않습니다. `domain_evidence`에는 평가가 실제로 인용한 기술 조사 근거만 원래 ID와 출처를 유지해 담습니다. 새로운 조사 결과를 만들어내지 않습니다.

`domain_eval`에는 두 기술의 여섯 항목씩 총 12개 평가, 전체 설명 글 `text`, `notes`가 들어갑니다. 평가 기준·점수·판단 이유는 `evaluations`에 유지합니다. 각 평가의 `evidence_ids`는 `domain_evidence`에 복사된 원래 조사 근거 ID를 가리킵니다. 조사 근거는 `source_checked` 상태이고 출처·인용문이 있어야 사용합니다. 근거가 없거나 확인에 실패한 항목은 `basis_status="unverified"`, `score=None`으로 남기고 `domain_evidence`에는 넣지 않습니다.

## 실행 조건

```bash
uv sync --locked --extra llm-openai
```

실제 평가에는 `LLM_PROVIDER=openai` 또는 `LM_PROVIDER=openai`, `LLM_MODEL`, `OPENAI_API_KEY`가 필요합니다. 설정은 실행 환경이나 작업 디렉터리에서 상위로 찾은 `.env`에서 읽습니다. 별도 설정 파일은 `DOMAIN_ENV_FILE`로 지정할 수 있습니다. `EMBEDDING_MODEL`은 기술 조사 단계의 설정이며 도메인 노드에서 사용하지 않습니다.

모델·키·provider 검증은 `common/config.py`의 공통 설정 함수를 사용하며 실행 환경 값이 파일보다 우선합니다. 도메인 실행부는 파일 탐색과 `DOMAIN_LLM_TIMEOUT_SECONDS`(기본 60초, 양의 유한한 숫자), 평가 한 번 동안의 설정 고정, 도메인 오류 분류를 담당합니다.

기술 조사 결과가 State에 들어온 뒤 그래프의 `domain` 노드가 `evaluate(state)`를 호출합니다. 이 폴더에는 단독 실행 CLI가 없습니다. 조사 결과가 없으면 모델을 호출하지 않고 미확인 평가를 반환합니다. 반면 모델 패키지·키·설정 오류는 명시적으로 알립니다.

## 점수와 근거

수치 항목은 원문이 직접 보고한 확정 백분율 또는 비교 가능한 기준값·적용값이 있고, 별도 근거 검토를 통과한 경우에만 코드에서 점수를 계산합니다. `최대`, `약`, 범위, 배수, 백분율포인트를 임의의 확정 백분율로 바꾸지 않습니다. 조건이 부족하면 판단 내용은 남길 수 있어도 점수는 보류합니다. 모델 품질과 운영 난이도는 원문 인용과 평가 기준에 대한 검토가 필요합니다.

입력 크기의 기본 한도는 평가·검토 요청 각각 131,072 UTF-8 바이트입니다. 두 기술의 근거가 한 요청에 들어가지 않으면 기술별로 나누고, 한 기술의 근거도 한도를 넘으면 그래프에서는 입력 예산 오류로 종료합니다. 이 평가는 공개 근거에 대한 판단이며 실제 GPU 운영 측정 결과를 뜻하지 않습니다.

## 응답 보정과 근거 부족

- 유효한 판단 없이 자료 부족을 명시하거나 정상적인 근거 검토가 판단을 지지하지 않는 항목은 `failure_kind="evidence_gap"`으로 반환하고 Supervisor가 필요한 조사를 결정합니다.
- 잘못된 인용·누락·측정값 형식은 `response_error`로 구분합니다. 오류 항목별 최대 한 번, 동일 근거와 검증 이유를 사용해 오류가 난 기술·항목만 재작성합니다. 성공한 항목은 보존합니다. 보정 결과도 같은 인용·측정값·독립 근거 검토를 통과해야 합니다.
- 재작성 후에도 응답 오류가 남으면 그래프는 `response_repair_exhausted`로 실패합니다. 이를 근거 부족으로 바꾸어 재검색하지 않습니다. 인증·연결·시간 초과 오류도 응답 보정 루프에 들어가지 않습니다.
- 근거 검토가 판단 자체는 지지하지만 정성 점수 기준은 지지하지 않을 때는 판단과 출처를 유지하고 점수만 `None`으로 둡니다. 정량 배수·범위는 검증된 서술로 보존할 수 있지만 임의의 확정 백분율·점수로 환산하지 않습니다.
- 내부 보정·검토 호출은 LangSmith의 하위 모델 호출로 남고 `domain_eval.notes`에 보정 여부가 기록됩니다. Supervisor의 노드 호출 수는 모델 호출 수와 다릅니다.

보정 요청은 한 번에 한 항목만 받는 동적 구조화 출력 스키마를 사용한다. 기술·항목·허용 근거 ID를 제한하고, measurement·score는 null로 고정한다. 모델은 인용문을 다시 쓰는 대신 원문 필드(claim 또는 excerpt)를 선택한다. 코드는 선택된 원문을 그대로 연결하고 기존 검토자가 판단의 지지 여부를 확인한다. 통과한 초기 정량 평가는 보정 대상에 포함하지 않는다.
