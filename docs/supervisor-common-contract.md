# Supervisor 리팩토링 공통 개발 계약

계약 버전: 1.2 구현 반영 · 적용 대상: KIVI와 InfiniGen 다관점 평가 프로젝트

`main-ver2`에 통합 구현한 Supervisor의 인터페이스와 동작을 설명한다. A·B·C 표기는 이후 팀 인계를 위한 책임 구분이며, 이번 구현은 담당 구분 없이 전체 범위를 반영했다. A는 흐름 제어, B는 조사·평가와 공통 State, C는 종합·보고서와 품질 평가를 뜻한다.

과제 기준은 [Multi-Agent Orchestration 안내](https://actually-war-1ea.notion.site/Multi-Agent-Orchestration-3d57f4c866938020a992fcc97e942ee6)다. 아래 타입·횟수·처리 방식은 이 프로젝트가 선택한 구현이며 과제 자체의 지정값은 아니다.

## 공통 원칙

1. 다음 에이전트는 Supervisor가 현재 State와 근거 충분성을 보고 선택한다. 순서를 고정한 목록을 차례로 실행하지 않는다.
2. 하위 에이전트는 결과를 Supervisor에 반환한다. 다른 에이전트를 직접 호출하지 않는다.
3. 초기 버전은 하위 에이전트를 한 번에 하나씩 실행한다. 필요한 에이전트를 상태에 따라 고르는 동작은 유지한다.
4. 기존 `Evidence`, `EvaluationResult`, `Synthesis`, `ReportDraft`를 재사용한다. 기존 필드의 의미를 담당자마다 다르게 바꾸지 않는다.
5. 작업 완료, 근거 부족, 실행 실패, 보고서 품질 통과를 구분한다.
6. 세 명이 계약을 검토하고 B가 공통 타입에 반영한다. 필드명·타입·의미를 바꾸면 이 문서와 관련 예시도 같은 변경에 포함한다.

## 담당과 수정 범위

아래 경로는 저장소 루트 기준이며 모두 구현되어 있다.

| 담당 | 소유 코드 | 책임 |
|---|---|---|
| A | `src/kv_cache_eval/graph/`, `features/supervisor/`, 실행·추적 설정 | 라우팅, 근거 충분성 결정, 재작업 배정, 종료·복구, 통합 |
| B | `src/kv_cache_eval/common/state.py`, `common/schemas.py`, `features/technical_research/`, `maturity/`, `market/`, `stakeholders/`, `domain/` | 공통 타입, 작업 범위 해석, 부분 결과 병합, 근거·평가, 오류 구분 |
| C | `src/kv_cache_eval/features/synthesis/`, `report/`, `quality/` | 종합, 보고서 초안·수정, 품질 판정, 최종 PDF |

각 담당자는 자기 기능의 테스트와 README 설명을 작성한다. A가 통합 테스트·그래프·트레이스를 정리한다. 공통 스키마를 수정해야 하는 작업은 B와 계약을 먼저 맞춘다.

## 노드 이름과 호출 방식

노드 이름은 아래 값으로 통일한다. 새 노드 이름은 세 명이 합의한 뒤 추가한다.

```python
from typing import Literal, TypedDict

AgentName = Literal[
    "technical_research", "maturity", "market", "stakeholders", "domain",
    "synthesis", "report", "quality", "export_pdf",
]
RunStatus = Literal["running", "completed", "incomplete", "failed"]
ExecutionStatus = Literal["completed", "needs_evidence", "failed"]
CheckStatus = Literal["pass", "fail", "unknown"]
```

기존 호출 형태인 `node(state: State) -> StateUpdate`를 유지한다. 입력 State는 직접 수정하지 않고 변경할 필드만 반환한다. `TypedDict`는 실행 중 값을 검증하지 않으므로 B는 요청·결과 검증 함수를 제공하고 A는 노드 실행 경계에서 이를 사용한다.

Supervisor는 `next_agent`를 설정하고 `add_conditional_edges`로 분기한다. 모든 하위 노드는 Supervisor로 돌아온다. `next_agent=None`은 실행 대상을 선택하지 않은 상태이며, 실행 종료 여부는 반드시 `status`와 함께 확인한다.

## State 필드

### 기존 작업 데이터

아래 필드를 유지한다. 초기값 `None`은 아직 생성·검토하지 않았다는 뜻이다. 빈 결과는 실제로 작업했지만 결과가 없다는 뜻이므로 구분한다.

| 필드 | 자료형 | 초기값 | 결과 작성 노드 |
|---|---|---|---|
| `selected_technologies` | 기존 기술 튜플 | KIVI, InfiniGen | 초기화 |
| `domain_and_criteria` | 기존 `DomainAndCriteria` | 기존 도메인·네 관점 | 초기화 |
| `kivi_evidence`, `infinigen_evidence` | `ResearchResult \| None` | `None` | 기술 조사 |
| `market_evidence` | `ResearchResult \| None` | `None` | 시장성 |
| `domain_evidence` | `ResearchResult \| None` | `None` | 도메인 |
| `maturity_eval` | `EvaluationResult \| None` | `None` | 성숙도 |
| `market_eval` | `EvaluationResult \| None` | `None` | 시장성 |
| `stakeholder_eval` | `EvaluationResult \| None` | `None` | 이해관계자 |
| `domain_eval` | `EvaluationResult \| None` | `None` | 도메인 |
| `evidence_gaps` | `list[EvidenceGap] \| None` | `None` | Supervisor의 근거 검사 |
| `synthesis` | `Synthesis \| None` | `None` | 종합 |
| `report` | `ReportDraft \| None` | `None` | 보고서 |

`score=None`은 미판단이며 낮은 점수나 0점으로 바꾸지 않는다. 근거의 출처·물리 PDF 페이지·원문 발췌·실험 조건·검증 상태를 보존한다.

### 추가할 결과와 제어 정보

State에서는 작업 데이터와 제어 정보를 주석으로 구분한다. 기존 결과 키를 중첩 객체로 옮기는 변경은 이번 계약에 포함하지 않는다.

| 필드 | 자료형 | 초기값 | 갱신 주체와 의미 |
|---|---|---|---|
| `quality_result` | `QualityResult \| None` | `None` | C의 품질 노드가 현재 초안을 검사한 결과 |
| `pdf_path` | `str \| None` | `None` | C의 내보내기 노드가 저장을 확인한 파일 경로 |
| `next_agent` | `AgentName \| None` | `None` | A가 선택한 다음 작업 |
| `retry_request` | `RetryRequest \| None` | `None` | A가 작성한 보완 요청. 최초 전체 작업이면 `None` |
| `completed_agents` | `list[AgentName]` | `[]` | A가 관리하는 현재 유효한 완료 결과 목록. 중복 금지 |
| `last_result` | `AgentExecution \| None` | `None` | 각 하위 노드가 마지막 실행 결과를 반환 |
| `evidence_decision` | `EvidenceDecision \| None` | `None` | A의 보고서 작성 가능 여부와 판단 이유 |
| `step_count` | `int` | `0` | A가 하위 노드 호출을 배정할 때마다 1 증가 |
| `max_steps` | `int` | `30` | 초기 전체 호출 상한 |
| `agent_calls` | `dict[AgentName, int]` | `{}` | A가 노드별 호출 횟수 관리 |
| `max_agent_calls` | `int` | `6` | 초기 노드별 호출 상한 |
| `report_revision` | `int` | `0` | 최초 초안은 0, 이후 보고서 재생성 배정 시 A가 1 증가 |
| `max_report_revisions` | `int` | `2` | 최초 초안 이후 추가 생성 상한 |
| `status` | `RunStatus` | `"running"` | A가 전체 실행 상태 관리 |
| `last_error` | `AgentError \| None` | `None` | A가 `last_result`에서 반영. 다음 작업 성공 시 해제 |
| `termination_reason` | `str \| None` | `None` | A가 완료·미완료·실패 사유 기록 |
| `trace_id` | `str` | 실행별 UUID | 초기화 후 불변. 체크포인트·외부 로그와 연결 |

30·6·2는 기본 설정값이다. 세 값은 설정으로 변경할 수 있으며 실제 성공 횟수를 보장하는 값이 아니다. 기존 `research_round`, `max_research_rounds`는 삭제했고 Supervisor 호출 예산으로 통일했다. 같은 기술·관점의 추가 검색은 한 번 수행한 뒤 재평가한다.

추가 제어 필드로 `question`(원래 질문), `pending_work`(노드별 재검토 요청), `gap_attempts`(기술·관점별 추가 검색 횟수), `decision_reason`(최신 라우팅 사유)을 유지한다. `AgentExecution.changed_keys`는 실제 바뀐 결과 키 목록이며 진단 notes만 바뀐 것은 근거 변경으로 계산하지 않는다.

## 재작업 요청과 실행 결과

다음 신규 자료형은 B가 `common/schemas.py`에 정의한다. `Technology`와 `EvidenceGap`은 기존 타입을 사용한다.

```python
class RetryRequest(TypedDict):
    technology: Technology | None
    criteria: list[str]
    reason: str

class AgentError(TypedDict):
    code: str
    message: str
    retryable: bool

class AgentExecution(TypedDict):
    agent: AgentName
    step: int
    status: ExecutionStatus
    gaps: list[EvidenceGap]
    error: AgentError | None
    changed_keys: list[str]

class EvidenceDecision(TypedDict):
    ready: bool
    reason: str
    blocking_gaps: list[EvidenceGap]
```

`RetryRequest.technology=None`은 두 기술 전체, `criteria=[]`는 해당 노드의 전체 항목을 뜻한다. `reason`은 빈 문자열을 허용하지 않는다. `retry_request`는 다음 한 번의 실행에만 적용하며 A가 다음 작업을 배정할 때 새 값 또는 `None`으로 교체한다.

`criteria`는 대상 노드의 기존 항목명을 사용한다. 도메인의 `처리량`, 시장성의 `시장 규모·성장성`처럼 각 모듈의 기준과 맞춰야 한다. 기술 조사에서는 기존 질문 ID인 `experiment_conditions`, `performance_results` 등을 사용한다. 관점 간 요청 변환은 A가 담당하고 B가 공통 매핑·허용 항목 검증을 제공한다. 알 수 없는 항목은 묵시적으로 전체 실행하지 않고 `invalid_request` 오류로 반환한다.

보고서 수정은 `criteria`에 기존 대목차 제목을 넣는다. 종합·품질 검사·PDF 저장은 전체 결과를 대상으로 하므로 `criteria=[]`를 사용한다. 보고서에 부분 수정 요청이 있어도 반환하는 `ReportDraft`에는 보존한 장을 포함한 전체 목차가 있어야 한다.

| 실행 상태 | 의미 | A의 처리 |
|---|---|---|
| `completed` | 요청한 범위의 작업을 수행함 | 결과의 유효성과 전체 범위 완료 여부를 검토 |
| `needs_evidence` | 실행했지만 판단 근거가 부족함 | `gaps`를 보고 필요한 조사·평가 선택 |
| `failed` | 요청을 수행하지 못함 | 오류 종류와 호출 예산을 보고 재시도 또는 종료 |

`needs_evidence`는 `gaps`를 한 건 이상 반환하고 `error=None`으로 둔다. `failed`는 `error`를 반드시 반환한다. `completed`도 검토한 한계나 공백을 `gaps`에 남길 수 있으나 보고서 작성 가능 여부는 Supervisor가 별도로 결정한다.

예상 가능한 API 오류는 구조화한다. `timeout`, `rate_limit` 등 일시적 오류만 제한적으로 재시도하고, `authentication`, `invalid_request`, `input_budget_exceeded`는 동일 요청을 반복하지 않는다. 예상하지 못한 예외는 공통 실행 경계에서 실패로 기록한다. State와 결정 로그에는 예외 클래스와 정규화한 오류 코드만 남기며, 요청 본문·API 키가 포함될 수 있는 원문 예외는 기록하지 않는다.

하위 노드는 자신의 결과 필드와 `last_result`만 반환한다. `last_result.step`은 입력의 `step_count`와 같아야 하며, Supervisor는 현재 배정한 노드·단계와 일치하는 결과만 처리한다. `updates`라는 별도 중첩 필드는 만들지 않는다.

### 도메인 재평가 요청 예시

```json
{
  "next_agent": "domain",
  "retry_request": {
    "technology": "InfiniGen",
    "criteria": ["처리량"],
    "reason": "추가 확보한 실험 조건과 비교 기준을 반영해 처리량 평가를 갱신한다."
  }
}
```

### 실행 실패 반환 예시

```json
{
  "last_result": {
    "agent": "domain",
    "step": 7,
    "status": "failed",
    "gaps": [],
    "error": {
      "code": "input_budget_exceeded",
      "message": "InfiniGen 평가 요청이 입력 한도를 초과했다.",
      "retryable": false
    },
    "changed_keys": []
  }
}
```

위 예시의 7은 예시 입력의 `step_count`다. 실패 시 기존 평가 결과를 삭제하거나 성공 결과처럼 덮어쓰지 않는다. 품질 검사를 정상 수행했지만 보고서가 미달인 경우에는 `last_result.status="completed"`, `quality_result.passed=false`를 반환한다.

## 결과 병합과 재검토 규칙

B의 조사·평가 노드는 부분 결과를 기존 결과와 병합한 뒤 자신이 소유한 State 필드를 반환한다. 동일 필드에 여러 노드가 동시에 쓰지 않는 현재 설계에서는 누적 reducer를 추가하지 않는다. 향후 병렬 실행을 추가할 때 공유 필드와 reducer 정책을 함께 바꾼다.

| 데이터 | 병합 기준 |
|---|---|
| 근거 | `id`로 중복 제거. 같은 ID의 주장·출처가 다르면 충돌로 처리 |
| 평가 | `(technology, criterion, stakeholder_group)`로 요청 범위만 갱신 |
| 도메인 인용 근거 | 병합된 전체 도메인 평가가 실제 인용한 근거로 다시 구성 |
| 주의사항 | 중복 제거한 현재 유효한 요약 유지. 전체 시도 로그를 계속 누적하지 않음 |
| 종합·보고서·품질 결과 | 최신 결과 하나로 교체 |

반환에 필드가 없으면 기존 값을 유지한다. `None`은 명시적인 무효화에 사용하고, 소유 노드가 임의로 다른 노드의 값을 지우지 않는다.

Supervisor는 결과가 실제로 변경됐을 때 관련 완료 표시를 해제한다. 새 결과의 존재만 확인하지 말고 변경 여부를 비교한다. 초기 구현은 아래처럼 보수적으로 처리한다.

| 변경된 데이터 | 재검토 대상 |
|---|---|
| 기술 조사 근거 | 성숙도·이해관계자·도메인 평가, 종합, 보고서, 품질, PDF. 자체 검색을 쓰는 완료된 시장성은 보존 |
| 시장 근거·시장성 평가 | 종합, 보고서, 품질, PDF |
| 그 외 관점 평가 | 종합, 보고서, 품질, PDF |
| 종합 | 보고서, 품질, PDF |
| 보고서 | 품질, PDF |

평가 재검토에서는 기존 값을 보존하되 `completed_agents`에서 제외해 오래된 결과임을 나타낸다. 종합·보고서·품질·PDF는 관련 값을 `None`으로 해제한다. 이 무효화는 A가 수행하는 예외적인 결과 필드 갱신이다. `report_revision`은 초안을 무효화해도 초기화하지 않는다.

근거·관점 평가가 바뀌면 관련 평가의 완료 상태를 해제하고 충분성을 다시 검사한다. 기존 근거 검사 결과는 진행 중 작업을 판단하는 과거 정보일 뿐 보고서 승인에 재사용하지 않는다. 현재 성숙도·이해관계자 평가는 기술 조사 근거를 입력으로 사용한다. 시장 근거 등 새로운 입력 의존성을 추가하면 위 재검토 표와 구현을 함께 확장한다.

`completed_agents`는 현재 요청 결과를 수락했다는 표시다. 부분 결과는 기존 결과와 병합하며 전체 범위의 누락·근거 부족은 별도 `evidence_decision`으로 점검한다. 완료 표시만으로 보고서 작성을 승인하지 않는다. 새 기술 근거가 추가되면 의존하는 관점에 해당 기술의 재검토 요청을 만들며, 근거가 그대로인 보완 검색 후에는 처음 부족했던 항목만 다시 평가한다.

## 근거 충분성과 라우팅

기존 근거 검사 기능을 A가 확장해 `evidence_gaps`와 `evidence_decision`을 갱신한다. `ready=true`는 네 관점이 검토됐고, 사용할 주장에 추적 가능한 근거가 있으며, 남은 공백을 보고서에서 미확인·한계로 명확히 다룰 수 있을 때만 허용한다.

중요 관점 전체가 실행 오류로 비어 있거나 핵심 주장에 출처가 없는 상태는 보고서 작성 가능 상태가 아니다. 미확인 점수를 만들거나 검색 실패를 근거 부재로 해석하지 않는다. 점수 없는 정성 평가도 판단 근거와 한계가 충분하면 유효한 결과가 될 수 있다.

| 상황 | 다음 행동 |
|---|---|
| 결과가 아직 없음 | 선행 근거가 준비된 해당 노드 선택 |
| 처리량의 실험 조건 부족 | 기술 조사에 `experiment_conditions` 요청 후 도메인 재평가 |
| 시장 채택 근거 부족 | 시장성에 `상용화·채택 현황` 보완 요청 |
| 일시적 API 오류 | 같은 요청을 남은 호출 예산 내에서 재시도 |
| 근거 검토 완료 | 종합 실행, 이후 보고서 생성 |
| 기존 평가가 보고서에 누락됨 | 보고서에 해당 내용을 반영하도록 요청 |
| 보고서 주장에 근거가 없음 | 근거 조사 또는 주장 수정 중 원인에 맞는 작업 선택 |
| 최신 보고서 품질 통과 | PDF 저장 |

종합을 선택하려면 `evidence_decision.ready=true`, 보고서를 선택하려면 유효한 종합이 있어야 한다. 모든 선행 조건과 종료 가드는 코드로 검증한다. 라우팅에 LLM을 쓰더라도 허용 노드·필수 선행 조건·호출 상한을 우회할 수 없다. 라우팅 방식은 과제가 특정 LLM 사용을 강제하지 않으므로 A가 규칙과 내용 판단을 조합해 구현한다.

## 보고서 품질 계약

C가 다음 판정 내용을 구현하고 B가 공통 타입을 반영한다.

```python
QualityCriterion = Literal[
    "groundedness", "neutrality", "bias_control", "perspective_coverage",
]

class QualityChecks(TypedDict):
    groundedness: CheckStatus
    neutrality: CheckStatus
    bias_control: CheckStatus
    perspective_coverage: CheckStatus

class QualityIssue(TypedDict):
    criterion: QualityCriterion
    section: str
    reason: str
    required_action: str

class QualityResult(TypedDict):
    report_revision: int
    passed: bool
    checks: QualityChecks
    reasons: dict[QualityCriterion, str]
    issues: list[QualityIssue]
```

`reasons`에는 네 항목의 판정 이유를 모두 기록한다. `fail` 또는 `unknown`인 항목에는 한 건 이상의 수정 지적을 포함한다. 네 항목이 전부 `pass`이고 수정 지적이 없을 때만 `passed=true`다. 네 관점의 제목이 존재한다는 이유만으로 내용 커버리지를 통과시키지 않는다.

| 항목 | 판정 내용 |
|---|---|
| `groundedness` | 인용 ID 존재 여부와 실제 주장·발췌의 대응, 출처·페이지 추적 가능성 |
| `neutrality` | 특정 기술의 무조건적 추천·우열 단정 여부 |
| `bias_control` | 단일 출처·유리한 결과 편중, 제약·반대 근거·검색 한계의 반영 |
| `perspective_coverage` | 두 기술의 성숙도·시장성·이해관계자·도메인 적용에 대한 판단·근거·한계 |

코드로 인용·목차 등 구조를 검사하고 LLM Judge로 내용을 검토하는 Hybrid 방식을 적용한다. 모델 호출·응답 파싱 실패는 `failed`로 반환한다. 정상 응답에서도 내용상 판정할 수 없으면 해당 항목을 `unknown`으로 둔다.

```json
{
  "report_revision": 1,
  "passed": false,
  "checks": {
    "groundedness": "pass",
    "neutrality": "pass",
    "bias_control": "pass",
    "perspective_coverage": "fail"
  },
  "reasons": {
    "groundedness": "검사한 주장과 출처 발췌의 대응이 확인됐다.",
    "neutrality": "조건을 명시한 비교이며 무조건적 추천이 없다.",
    "bias_control": "두 기술의 제약과 공개 정보의 한계를 함께 설명한다.",
    "perspective_coverage": "이해관계자별 편익과 부담이 빠져 있다."
  },
  "issues": [{
    "criterion": "perspective_coverage",
    "section": "4. 관점별 평가 결과",
    "reason": "운영자와 개발자의 편익·부담이 구분되지 않았다.",
    "required_action": "기존 stakeholder_eval의 근거를 사용해 해당 내용을 보완한다."
  }]
}
```

위 판정은 연동을 위한 예시이며 실제 실행 결과가 아니다. 보고서 노드는 재작성 시 입력 State의 `quality_result.issues`를 읽는다. A는 이전 판정을 요청 입력으로 보존하고, 새 보고서가 반환된 뒤 이전 `quality_result`를 해제한다. 품질 노드는 새 `report_revision`을 검사 결과에 복사한다.

보고서 재생성을 배정할 때 A는 `report`, `quality`, `export_pdf`의 완료 표시를 해제하고 `pdf_path=None`으로 둔다. 이전 초안과 판정은 수정 입력으로만 사용한다. 생성이 실패했다면 새 버전의 보고서가 완료되기 전까지 품질 검사·PDF 저장을 배정하지 않는다. 품질 검사는 현재 보고서의 완료 표시가 있어야 하고, PDF 저장은 현재 품질 검사의 완료 표시까지 있어야 한다.

보고서는 기존 대목차를 유지하고 각 관점을 판단·근거·실험 또는 적용 조건·한계로 설명한다. 최소 페이지 수를 채우기 위한 내용 추가는 하지 않는다. `SUMMARY`, `REFERENCE`를 포함하며 최종 PDF는 참고문헌을 포함해 최대 10쪽으로 제한한다.

## 횟수와 종료 규칙

A는 노드를 배정하기 전에 `step_count`와 해당 `agent_calls`를 증가시킨다. 실패한 호출과 품질 검사·PDF 저장도 호출 수에 포함한다. 재개 시 이미 배정돼 체크포인트에 기록된 동일 실행은 다시 배정하지 않는다. 새로운 재시도는 새 호출로 계산한다.

최초 보고서 호출은 `report_revision=0`이다. 이후 보고서 재생성을 배정할 때마다 1 증가시키며 실패한 생성 시도도 포함한다. 과거 보고서 필드를 지웠더라도 호출 이력으로 최초 생성 여부를 판정한다. 상한을 초과하는 작업은 배정하지 않는다. 현재 결과를 검토하고 종료할 Supervisor 실행은 하위 호출 수에 포함하지 않는다.

| 전체 상태 | 종료 조건 |
|---|---|
| `running` | 다음 작업을 진행할 수 있음 |
| `completed` | 최신 보고서 품질 통과, PDF 저장·파일 확인·10쪽 이하 확인 완료 |
| `incomplete` | 근거·품질 미달 상태로 상한에 도달하거나 추가 작업의 진전이 없어 종료 |
| `failed` | 인증·설정·스키마 오류 등 복구 불가, 또는 실행 오류 재시도 소진 |

호출 예산이 마지막까지 사용됐어도 방금 반환된 결과가 모든 완료 조건을 충족하면 성공으로 종료할 수 있다. 상한 자체를 품질 통과로 해석하지 않는다. 모든 종료에서 `next_agent=None`과 구체적인 `termination_reason`을 기록한다.

`export_pdf`는 품질 판정의 버전이 현재 보고서와 일치하고 `passed=true`일 때만 실행한다. 페이지 초과는 `failed`와 `report_too_long` 오류로 반환한다. A는 보고서 수정 예산이 남아 있으면 압축을 요청하고 새 초안을 다시 품질 검사한다. 파일 쓰기 오류는 별도 실행 오류로 처리한다. 검사 전 파일은 임시 경로에 만들고, 조건을 충족한 파일만 최종 경로로 옮겨 `pdf_path`를 반환한다.

## 로그와 재개

State에는 최신 결과와 제어에 필요한 값만 둔다. 외부 기록에는 `trace_id`, `step`, 노드, 결정, 이유, 요청 범위, 실행 상태, 시각을 남긴다. LangSmith 실행 메타데이터에 동일한 `trace_id`를 기록하고, 그래프 체크포인트의 `thread_id`도 이 값과 연결한다.

A는 지속 가능한 체크포인트 저장과 같은 실행 ID를 사용한 재개 경로를 구현한다. 완료된 노드를 처음부터 다시 수행하는 것은 재개 검증으로 인정하지 않는다. 중단 시 외부 API 응답이 저장되지 않았다면 해당 호출이 다시 발생할 수 있으므로 정확히 한 번 실행을 보장한다고 표현하지 않는다. PDF 저장은 같은 실행의 안정적인 경로와 임시 파일 교체 방식으로 중복 최종 파일을 방지한다.

추적 ID와 상태 필드의 존재만으로 복구가 구현된 것으로 간주하지 않는다. 실제 중단·재개 시나리오로 확인한다.

## 통합 시나리오와 인계 기준

| 시나리오 | 기대 결과 | 주 담당 |
|---|---|---|
| 최초 조사·평가 | 필요한 노드를 선택하고 결과를 State에 반영 | A·B |
| InfiniGen 처리량만 보완 | 해당 기술·항목을 처리하고 기존 KIVI 결과 보존 | B |
| 입력 한도 초과 | 실행 실패로 반환하고 동일 검색을 반복하지 않음 | A·B |
| 일시적 오류 | 호출 상한 안에서 재시도 후 성공 또는 명시적 실패 | A |
| 근거 변경 | 영향받는 결과의 완료 표시와 오래된 품질 통과 해제 | A·B |
| 보고서 내용 누락 | 품질 미달 → 수정 → 재검사, 최초 판정 재사용 금지 | A·C |
| API 오류 없는 품질 미달 | quality 실행은 completed, 보고서 passed는 false | C |
| 보고서 10쪽 초과 | 수정 요청 후 새 버전 품질·페이지 재검사 | A·C |
| 상한 도달 | 성공으로 위장하지 않고 종료 사유 기록 | A |
| 중단 후 재개 | 저장된 결과를 사용해 남은 작업부터 진행 | A |

각 담당자는 노드 호출 방법, 대표 입력, 예상 반환값, 수행한 테스트를 인계한다. 외부 응답을 대체한 테스트와 실제 API 실행 결과를 구분한다. 통합 후 실제 실행으로 동적 라우팅·재작업 경로, 최종 PDF, LangSmith 트레이스를 확인한다.

개발 순서는 공통 타입 반영 → A·B·C 개별 브랜치 작업 → 정상 흐름 통합 → 재작업·실패·복구 검증 → 실제 실행이다. README에는 담당별 구현, Supervisor 선정 이유, State 설계 근거, 실행 방법을 반영한다. 제출물은 구분된 GitHub 브랜치, LangSmith 트레이스 이미지, 최대 10쪽 보고서다.

## 도메인 응답 보정 계약 (1.2)

`Evaluation.failure_kind`는 미확인 도메인 항목의 선택 필드이며 `evidence_gap`과 `response_error`를 구분한다. 도메인은 오류 항목별 한 번만 기존 근거로 오류 항목을 다시 작성·검토한다. 응답 오류가 소진되면 공통 실행 경계는 `response_repair_exhausted`, `retryable=false`로 반환하고 Supervisor는 실패 종료한다. 정상적인 자료 부족만 재조사 대상으로 남는다. 정성 판단 검증과 점수 기준 검증이 분리돼 판단은 유지하면서 점수만 null로 보류할 수 있다.

보정 요청은 한 번에 한 항목만 받는 동적 구조화 출력 스키마를 사용한다. 기술·항목·허용 근거 ID를 제한하고, measurement·score는 null로 고정한다. 모델은 인용문을 다시 쓰는 대신 원문 필드(claim 또는 excerpt)를 선택한다. 코드는 선택된 원문을 그대로 연결하고 기존 검토자가 판단의 지지 여부를 확인한다. 통과한 초기 정량 평가는 보정 대상에 포함하지 않는다.

정상적인 검토 응답에서 `supported=false`는 해당 판단을 폐기하고 `evidence_gap`으로 반환한다. 인용문 일치만으로 판단의 타당성을 통과시키지 않는다. 검토 결과 누락·중복·타입 오류는 `response_error`로 보정한다.
