# KIVI·InfiniGen 공통 기술 조사

Graph의 단일 `technical_research` 노드는 `node.py`의 `research(state, user_question=...)`에 연결됩니다. 이 함수가 선정된 두 기술에 같은 `research_technology(state, technology)`를 적용하고 `kivi_evidence`와 `infinigen_evidence`를 함께 반환합니다. 질문, RAG/Tavily 검색, 본문 검토, 구조화 추출, 제한된 재검색 흐름은 공통입니다.

## 원문과 인덱스

`sources/kivi.json`은 KIVI 원문 15쪽과 KVQuant 독립 평가 34쪽을, `sources/infinigen.json`은 InfiniGen 원문 18쪽과 FlexGen 23쪽·H2O 50쪽을 선언합니다. PDF 5개는 `data/documents/`에 놓습니다. `ingest.py`는 파일 SHA256, 실제 PDF 페이지, 기술별 100쪽과 전체 200쪽 제한을 검사합니다. 현재 합계는 140쪽입니다. KVQuant 파일의 온라인 버전은 확인되지 않아 URL을 비워 두었습니다.

`retriever.py`는 검증된 PDF를 물리 페이지별로 분할하고 BGE-M3 임베딩과 FAISS로 검색합니다. 인덱스는 기술별로 `data/indexes/kivi/`, `data/indexes/infinigen/`에 저장하며 원문·설정·출처 메타데이터 지문이 다르면 다시 만듭니다. 로컬 FAISS 캐시의 pickle은 생성 주체, 지문, 파일 해시와 문서 메타데이터를 확인한 뒤 읽습니다. 기존 InfiniGen 색인은 `data/indexes/infinigen_before_shared/`에 보존했습니다.

## 질문과 근거

`prompts.py`의 공통 질문 10개는 문제, 원리, 구현, 실험 조건, 성능, 품질, 한계, 독립 평가, 현재 공개 상태, 논문과 공개 구현 연결을 다룹니다. 앞의 논문 질문은 RAG, 공개 상태는 웹, 연결은 둘 다 검색합니다. 이전 평가의 근거 공백은 원래 기준에 맞는 범주로 다시 질문합니다. 경로와 문서 제목은 검색 단서이며 본문 사실의 증거가 아닙니다.

`main.py`는 `graph.run(QUESTION)`만 호출합니다. 이 요청은 단일 Graph 조사 노드에서 `research_technology(..., user_question=QUESTION)`로 전달되고, 각 기술의 공통 세부 질문에 붙어 검색과 LLM 검토에 사용됩니다. 사용자 질문의 기술 분류나 주장은 검증할 맥락일 뿐 근거로 취급하지 않습니다. 기본 호출은 기존 10개 질문과 경로를 그대로 사용합니다.

웹 경로는 `TavilySearch` 한 번으로 URL과 `raw_content`를 받아 문서 후보를 만듭니다. 검색 요약 `content`는 근거로 사용하지 않으며, 본문이 없는 결과는 미확인으로 남깁니다.

`workflow.py`는 질문별 검색 → 충분성 검토 → 최대 3개 주장 추출을 수행합니다. 근거에는 실제 검색 청크의 ID, 문서, 물리 PDF 페이지 또는 URL, 발췌를 연결합니다. 실험 모델·워크로드·비교 기준은 같은 청크 본문에 있을 때만 저장합니다. KIVI의 문서 파일명과 `:principle:` 등 근거 ID 범주는 기존 TRL 평가에서 사용하는 형태를 유지합니다. KVQuant 문서가 KIVI를 명시적으로 평가하는 경우에만, 평가 주체를 분명히 밝힌 주장으로 연결합니다. FlexGen·H2O 등 배경 문서의 자체 결과를 대상 기술의 결과로 귀속하지 않습니다.

검색 근거가 부족하거나 발췌 검증에 실패하면 시도 횟수 안에서 재검색하고, LLM 호출 상한에 도달하면 미확인으로 남깁니다. `source_checked`는 발췌와 출처가 연결되었다는 뜻입니다. 주장 전체의 의미, 웹 사이트의 공식성, 상용 채택 또는 TRL을 확정하지 않습니다.

## 실행

```bash
LANGSMITH_TRACING=false .venv/bin/python scripts/run_research.py index --technology KIVI
LANGSMITH_TRACING=false .venv/bin/python scripts/run_research.py index --technology InfiniGen
LANGSMITH_TRACING=false .venv/bin/python scripts/run_research.py research --technology KIVI --question-id principle --max-attempts 1 --max-llm-calls 2 --output-name research_result_shared_check.json
```

임베딩 모델이 이미 캐시된 환경에서만 명령 앞에 `HF_HUB_OFFLINE=1`을 붙여 오프라인으로 실행할 수 있습니다. `research`는 `.env` 또는 셸의 `LLM_PROVIDER`, `LLM_MODEL`, API 키를 사용합니다. 현재 생성 제공자는 OpenAI입니다. `--output-name`은 각 기술의 `data/cache/<technology>/` 안에서 기존 결과와 겹치지 않게 지정합니다. InfiniGen 전용 기존 고정 질문집 평가는 `evaluate --technology InfiniGen`으로 계속 실행합니다.

이전 InfiniGen 웹 경로 제한 실행은 Tavily 검색과 2개 URL의 본문 추출까지 확인했지만, `public_status` 질문의 공개 구현·지원·채택은 충분히 확인하지 못해 미확인으로 남겼습니다. 원시 기록은 `data/cache/infinigen/research_result_web_live*.json`에 보존되어 있습니다. 이는 채택 사례가 없다는 결론이 아닙니다.
