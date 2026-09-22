# InfiniGen 기술 조사: 논문 RAG와 웹 보완

선정 기술은 InfiniGen으로 고정되어 있습니다. `prompts.py`의 공통 질문은 목적에 따라 `rag`, `web`, `both` 중 한 경로를 선택합니다. 원리·실험·조건은 논문 RAG, 현재 공개 구현·지원·채택은 웹, 논문과 공개 구현의 연결은 둘 다 검색합니다. RAG 질문의 근거가 부족하면 최대 시도 횟수 안에서 다음 검색을 `both`로 넓힙니다. 이 선택은 **어디서 찾을지** 정할 뿐 사실을 증명하지 않습니다.

`node.py`는 `research_infinigen(state)`에서 `infinigen_evidence`만 반환합니다. PDF 검색은 실제로 필요할 때 `retriever.py`의 로컬 BGE-M3/FAISS 색인을 로딩하므로 웹 전용 질문은 그 비용이 없습니다. `web.py`는 `TavilySearch`로 최대 3개 URL을 찾고 `TavilyExtract`가 실제 추출한 본문만 LangChain `Document`로 전달합니다. 검색 요약·답변은 근거로 쓰지 않으며, 웹 본문은 색인 PDF나 영구 RAG 코퍼스에 추가하지 않습니다. 웹 본문은 URL당 최대 6000자로 제한합니다.

`workflow.py`의 LangGraph는 검색 → 본문 충분성 검토 → 구조화 추출을 최대 설정 횟수까지 실행합니다. 재검색 때 이미 찾은 PDF 문서를 유지합니다. LLM 호출 상한도 별도로 적용합니다. API 키 누락·검색 실패·일부 URL 본문 추출 실패는 `notes`에 남기며, 실패 자체를 근거 부재로 판정하지 않습니다.

PDF 근거는 원문 청크 ID·출처 ID·물리 페이지·발췌를, 웹 근거는 청크 ID·검색 도구의 URL·`page=None`·추출 본문 발췌를 확인합니다. 웹은 `web_external`로 기록하며 원논문이나 공식 출처로 자동 승격하지 않습니다. 웹의 `source_checked`는 **해당 URL의 추출 본문과 인용을 연결했다는 뜻**입니다. 공식성, 실제 상용 채택, 현재 유효성, TRL을 확정하는 표시는 아닙니다. 모델·워크로드·비교 기준도 인용한 바로 그 본문에 있는 값만 저장합니다.

```bash
# .env에는 사용자가 정한 생성 모델·키와 TAVILY_API_KEY가 있어야 함
# 이전 결과를 보존하려면 매번 다른 --output-name을 사용
env -u OPENAI_API_KEY LANGSMITH_TRACING=false .venv/bin/python scripts/run_infinigen.py research \
  --question-id public_status --max-attempts 1 --max-llm-calls 2 \
  --output-name research_result_web_check.json
```

2026-09-22 실제 Tavily 검색·본문 추출은 2개 URL에서 성공했습니다. `public_status` 질문의 제한 실행 두 번은 웹 본문에서 공개 구현·지원·채택을 충분히 확인하지 못해 미확인으로 남겼습니다(`data/cache/infinigen/research_result_web_live*.json`). 이 결과는 웹 경로와 거절 동작의 확인이며 채택 사례가 없다는 결론은 아닙니다. 기존 PDF 검색 평가와 원시 결과는 변경하지 않았습니다.
