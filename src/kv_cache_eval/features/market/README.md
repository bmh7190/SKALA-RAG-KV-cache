# 시장성 평가 모듈

KIVI와 InfiniGen의 시장 규모·성장성, 상용화·채택 현황, 생태계 지지를 설계서의 기준으로 평가합니다.

## 구성

- `rubric.py`: 설계서의 1~5점 채점 규칙
- `research.py`: 시장·채택·생태계 검색과 URL 정규화
- `analysis.py`: 인용 URL 검증, 공통 Evidence 및 Evaluation 변환
- `node.py`: Tavily 검색과 구조화 LLM 판단을 연결하는 LangGraph 노드
- `cli.py`: 전체 그래프와 별개로 시장성 노드만 실행하는 명령

## 실행 설정

현재 실제 실행은 OpenAI 생성 모델과 Tavily 검색을 지원합니다. 저장소 루트의 `.env`에 다음 값을 설정합니다. 실제 키는 Git에 커밋하지 않습니다.

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=<팀에서 선택한 모델명>
OPENAI_API_KEY=<로컬 키>
TAVILY_API_KEY=<로컬 키>
```

API를 호출하지 않고 설정만 확인합니다.

```bash
uv run --locked kv-market-eval --check-config
```

설정 확인 후 시장성 조사·평가를 실행합니다.

```bash
uv run --locked kv-market-eval
```

결과 JSON은 `market_evidence`와 `market_eval`을 포함합니다. 검색 요약문만 확보된 자료는 `unverified`로 유지하고, 원문까지 확보된 자료만 `source_checked`로 처리합니다. 검색 결과에 없는 URL을 구조화 판단이 인용하면 실행을 중단합니다.
