# main ver2 작업 기록

## 목표와 기준

- 시작 브랜치와 커밋: `main`, `5df0f7a`.
- 작업 브랜치: `main-ver2`.
- Supervisor의 상태 기반 라우팅, 선택적 재작업, 보고서 품질 평가·수정, 종료·재개를 구현한다.
- LangGraph의 StateGraph·조건부 edge·체크포인트와 LangChain의 구조화 출력을 사용한다.
- 기존 근거 출처와 평가 기준을 보존하며 공통 설정·근거 수집·노드 경계의 중복을 줄인다.
- [공통 계약](supervisor-common-contract.md)을 구현과 함께 갱신한다. API 키·실행 원문·체크포인트는 커밋하지 않는다.

## 작업 단위

1. 기준 기록과 공통 계약 보관.
2. State·공통 런타임·작업 범위 및 부분 결과 병합.
3. 보고서 초안·품질 평가·최종 PDF 분리.
4. Supervisor·동적 그래프·체크포인트와 추적.
5. 실패·재작업·복구 검증, 실행 문서와 결과 기록.

## 기준 검증

- 2026-10-07: 기존 unittest 64개 실행, 실패 1개·오류 1개.
- `test_rejects_hallucinated_source_url`: 검색 결과 밖 URL을 거부하지 않아 실패.
- `test_default_function_node_reports_unimplemented`: 구현된 synthesis를 미구현으로 가정한 테스트가 실제 API에 연결하려다 실패.
- 별도 offline guard 소스가 없으므로 테스트 통신 차단을 먼저 명시적으로 추가한다.
- `langgraph 1.2.12`, `langchain-core 1.6.4`, `langchain-openai 1.6.3` 확인. SQLite checkpointer 추가 필요.
- 조사한 공식 API: [Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api), [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents).

## 구현과 검증 기록

이하 항목은 각 작업 커밋에서 변경 내용·검증 결과·남은 문제를 추가한다.

### 공통 계약과 선택적 작업

- State에 제어·품질·추적 필드를 추가하고 공통 모델 팩토리, 근거 수집·인용 검증, 작업 범위·부분 결과 병합을 분리했다.
- 기술 재조사는 요청된 질문만 실행한다. 성숙도·이해관계자·도메인 평가는 요청 기술·항목을 처리하고 다른 결과를 보존한다.
- 도메인 입력 예산을 128KiB로 조정하고 그래프 실행 중 입력·호출 오류를 근거 부족으로 삼키지 않도록 했다.
- 검색 결과 밖 시장 URL은 검증 오류로 복구했다. SQLite 체크포인터 의존성을 잠금 파일에 반영했다.
- 검증: 기존 도메인·시장성·이해관계자 테스트 29개와 신규 범위·병합·근거 충돌 테스트 4개 통과. 전체 그래프 테스트는 구조 변경 단계에서 교체 예정.
