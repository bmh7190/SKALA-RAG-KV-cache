# 도메인 적용 평가

GPU 기반 클라우드 LLM 서비스에 KIVI와 InfiniGen을 적용할 때의 메모리, 전송량, 지연시간, 처리량, 모델 품질, 운영 난이도를 평가합니다. 평가 기준과 점수 구간은 `rubric.py`가 정의합니다.

## State 연결

`node.evaluate(state)`는 `domain_and_criteria["domain"]`, `kivi_evidence`, `infinigen_evidence`를 읽고 `domain_evidence`와 `domain_eval`만 반환합니다. `market_evidence`나 `market_eval`은 읽거나 수정하지 않습니다. `domain_evidence`에는 평가가 실제로 인용한 기술 조사 근거만 원래 ID와 출처를 유지해 담습니다. 새로운 조사 결과를 만들어내지 않습니다.

`domain_eval`에는 두 기술의 여섯 항목씩 총 12개 평가, 전체 설명 글 `text`, `notes`가 들어갑니다. 평가 기준·점수·판단 이유는 `evaluations`에 유지합니다. 각 평가의 `evidence_ids`는 `domain_evidence`에 복사된 원래 조사 근거 ID를 가리킵니다. 조사 근거는 `source_checked` 상태이고 출처·인용문이 있어야 사용합니다. 근거가 없거나 확인에 실패한 항목은 `basis_status="unverified"`, `score=None`으로 남기고 `domain_evidence`에는 넣지 않습니다.

## 실행 조건

```bash
uv sync --locked --extra llm-openai
```

실제 평가에는 `LLM_PROVIDER=openai` 또는 `LM_PROVIDER=openai`, `LLM_MODEL`, `OPENAI_API_KEY`가 필요합니다. 설정은 실행 환경이나 작업 디렉터리에서 상위로 찾은 `.env`에서 읽습니다. 별도 설정 파일은 `DOMAIN_ENV_FILE`로 지정할 수 있습니다. `EMBEDDING_MODEL`은 기술 조사 단계의 설정이며 도메인 노드에서 사용하지 않습니다.

기술 조사 결과가 State에 들어온 뒤 그래프의 `domain` 노드가 `evaluate(state)`를 호출합니다. 이 폴더에는 단독 실행 CLI가 없습니다. 조사 결과가 없으면 모델을 호출하지 않고 미확인 평가를 반환합니다. 반면 모델 패키지·키·설정 오류는 명시적으로 알립니다.

## 점수와 근거

수치 항목은 원문이 직접 보고한 확정 백분율 또는 비교 가능한 기준값·적용값이 있고, 별도 근거 검토를 통과한 경우에만 코드에서 점수를 계산합니다. `최대`, `약`, 범위, 배수, 백분율포인트를 임의의 확정 백분율로 바꾸지 않습니다. 조건이 부족하면 판단 내용은 남길 수 있어도 점수는 보류합니다. 모델 품질과 운영 난이도는 원문 인용과 평가 기준에 대한 검토가 필요합니다.

입력 크기의 기본 한도는 평가·검토 요청 각각 32,000 UTF-8 바이트입니다. 두 기술의 근거가 한 요청에 들어가지 않으면 기술별로 나누고, 한 기술의 근거도 한도를 넘으면 해당 기술의 평가를 미확인으로 반환합니다. 이 평가는 공개 근거에 대한 판단이며 실제 GPU 운영 측정 결과를 뜻하지 않습니다.
