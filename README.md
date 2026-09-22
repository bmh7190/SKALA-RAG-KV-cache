# KV Cache 최적화 기술 평가: 팀 협업 시작 구조

GPU 기반 클라우드 LLM 서비스에 KIVI(KV Cache 양자화)와 InfiniGen(CPU Host Memory에 KV를 두고 필요한 데이터를 GPU로 가져오는 접근)을 적용할 때의 근거와 조건을 비교하는 LangGraph 구조입니다. 기술은 사람이 선정했습니다. 알고리즘 구현이나 GPU 벤치마크가 아니라 공개 자료에 기반한 기술 평가가 목적입니다. 단일 승자보다 관점별 차이, 상충 관계, 적용 조건을 보고합니다.

## 현재 상태

- **구현됨:** 공유 State·자료형·초기값, 구조적 근거 공백 검사, 재조사 횟수 라우팅, 병렬 조사/평가 합류 그래프, 오프라인 smoke test, InfiniGen 전용 로컬 PDF RAG와 제한된 재검색 흐름.
- **TODO:** KIVI 조사, 네 평가 노드, 내용 타당성 검사, 종합, 보고서 및 PDF 생성. KIVI와 평가 등 기본 기능 노드는 `NotImplementedError`로 중단됩니다. 테스트 대체 노드의 빈 데이터와 문구는 연구 결과가 아닙니다.
- InfiniGen 임베딩은 사용자 지정 `BAAI/bge-m3`를 사용합니다. 생성 모델은 `.env`의 `LLM_PROVIDER`와 `LLM_MODEL`에서 읽으며 저장소에서 모델명을 고정하지 않습니다.

## 설치와 실행

Python **3.11**을 사용합니다(`.python-version`, `pyproject.toml`). 현재 잠금 파일과 프로젝트 가상환경 설치는 macOS ARM64의 Python 3.11.15에서 검증했습니다. `uv`가 없으면 [공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)를 따라 설치하세요. Python 3.11이 없으면 `uv python install 3.11`로 준비할 수 있습니다([공식 Python 안내](https://docs.astral.sh/uv/concepts/python-versions/)).

```bash
# 현재 공통 State/Graph 작업에 필요한 기본 설치
uv sync --locked

# 팀 전체 개발 도구를 한 번에 설치: RAG, 웹 검색, 보고서, 선택형 OpenAI 연동
uv sync --locked --all-extras

# 오프라인 단위 테스트 및 그래프 컴파일 확인
HF_HUB_OFFLINE=1 uv run --locked python -m unittest discover -s tests -v
uv run --locked python -c 'from kv_cache_eval.graph import build_graph; print(build_graph())'
```

`pyproject.toml`이 직접 의존성의 단일 기준이고 `uv.lock`이 해결된 버전을 고정합니다. `--locked`는 두 파일이 맞지 않으면 설치를 중단합니다([uv 공식 문서](https://docs.astral.sh/uv/concepts/projects/sync/)). 필요한 기능만 설치하려면 `uv sync --locked --extra rag`처럼 선택할 수 있습니다.

| 설치 범위 | 패키지와 용도 |
| --- | --- |
| 기본 | `langgraph`: 현재 구현된 StateGraph 배선과 테스트에 실제 사용. `python-dotenv`: 명시적 `.env` 로딩 함수에 사용 |
| `rag` | `langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `sentence-transformers`, `faiss-cpu`, `pypdf`, `pdfplumber`: InfiniGen PDF 로딩·분할·오픈소스 임베딩·로컬 검색용 |
| `web` | `langchain-tavily`: 후속 웹 검색용 |
| `report` | `reportlab`: 후속 PDF 생성용 |
| `llm-openai` | InfiniGen 조사에서 `LLM_PROVIDER=openai`로 선택할 때 사용할 클라이언트. 모델의 기본값은 없음 |

전체 옵션 설치는 패키지만 준비합니다. 모델 가중치 다운로드, 유료 API 호출, 문서 인덱싱은 수행하지 않습니다. 그래프 **컴파일**은 가능하지만 아직 구현되지 않은 KIVI 노드에서 기본 실행이 중단됩니다. 단위 테스트는 키·원문·네트워크가 필요 없습니다.

### 환경변수와 필요한 키

```bash
cp .env.example .env
# .env를 열어 사용할 기능에 필요한 값만 직접 입력
```

| 변수 | 언제 필요한가 |
| --- | --- |
| `LLM_PROVIDER`, `LLM_MODEL` | InfiniGen 조사를 실제 실행할 때 사용. 현재 구현된 제공자는 `openai`이며 모델명은 사용자가 설정 |
| `EMBEDDING_MODEL` | 사용자 지정 `BAAI/bge-m3`. 로컬 dense 임베딩이므로 별도 유료 임베딩 API 키나 벡터 DB 키는 필요 없음 |
| `OPENAI_API_KEY` | OpenAI 생성 LLM을 실제 호출할 때만 필요 |
| `TAVILY_API_KEY` | Tavily 웹 검색을 구현하고 호출할 때만 필요 |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | 추적은 기본 `false`. 사용하기로 선택한 경우에만 `true`와 키·프로젝트 설정 |

`BAAI/bge-m3`는 [공개 모델 카드](https://huggingface.co/BAAI/bge-m3)의 일반 로컬 다운로드에 `HF_TOKEN`이 필수가 아니므로 예시에 넣지 않았습니다. InfiniGen RAG는 `HuggingFaceEmbeddings`와 로컬 FAISS의 **dense 벡터 검색**을 사용합니다. 모델의 sparse·multi-vector 기능은 사용하지 않습니다.

`.env`는 파일만 만든다고 자동 적용되지 않습니다. InfiniGen CLI와 노드는 클라이언트를 만들기 전에 `load_environment()`를 호출합니다. 별도 진입점을 만들 때도 다음처럼 호출합니다.

```python
from kv_cache_eval.common.config import get_embedding_model, load_environment

load_environment()  # 현재 작업 디렉터리의 .env; 다른 위치라면 경로를 인수로 전달
embedding_model = get_embedding_model()
```

`load_environment()`는 이미 셸에 설정된 값을 덮어쓰지 않습니다. 셸 값과 `.env` 값이 다르면 셸 값이 우선하므로 실행 환경을 확인하세요. `.env`는 Git에서 무시됩니다. `index`와 `evaluate`는 생성 LLM API를 호출하지 않으며 `research`와 InfiniGen 노드는 호출합니다.

## 그래프

실선은 현재 배선 및 구조적 검사입니다. `TODO` 표시 노드는 함수 틀만 있습니다. 근거 부족 시 최대 `max_research_rounds`번 추가 조사하고 네 평가를 다시 실행합니다. 횟수를 소진하면 `evidence_gaps`를 유지한 채 종합 노드로 전달합니다. 종합 담당자는 미해결 항목을 명시해야 합니다.

```mermaid
flowchart TD
    I[입력 확인] --> K[KIVI 조사 TODO]
    I --> F[InfiniGen 조사 RAG]
    K --> J{{두 조사 완료}}
    F --> J
    J --> T[TRL TODO]
    J --> M[시장성 TODO]
    J --> S[이해관계자 TODO]
    J --> D[도메인 TODO]
    T --> G{{네 평가 완료}}
    M --> G
    S --> G
    D --> G
    G --> C[근거 구조 검사]
    C -->|공백 있고 횟수 남음| R[재조사 횟수 증가]
    R --> K
    R --> F
    C -->|공백 없음 또는 횟수 소진| Y[종합 TODO]
    Y --> P[보고서 TODO]
    P --> E[종료]
```

`check_evidence`는 결과 누락, 미확인 출처, 근거 ID 연결을 확인하는 **최소 구조 검사**입니다. 문서의 실제 신뢰성, 실험 조건의 비교 가능성, 주장 내용의 타당성 판정은 조사/평가 담당자가 추가해야 합니다. 빈 근거나 빈 평가는 통과하지 않습니다.

## KIVI·InfiniGen 공통 기술 조사

두 기술은 `research_technology(state, technology)`의 같은 질문·검색·검토·추출·제한된 재검색 흐름을 사용합니다. `research_kivi`와 `research_infinigen`은 각각 해당 State 키만 반환합니다. 상세 설계와 인용 규칙은 [`technical_research/README.md`](src/kv_cache_eval/features/technical_research/README.md)를 참조하세요.

`technical_research/sources/`의 기술별 manifest가 PDF 버전·SHA256·물리 페이지·역할을 정의합니다. 현재 원문은 `data/documents/`에 5개 PDF로 저장되어 있으며, KIVI 49쪽과 InfiniGen 91쪽, 총 140쪽입니다. 색인 전 실제 파일의 해시와 페이지 예산을 확인합니다. 기술별 BGE-M3/FAISS 색인은 `data/indexes/kivi/`와 `data/indexes/infinigen/`에 분리됩니다.

```bash
# 원문이 data/documents/에 준비된 상태에서 색인 또는 재사용
LANGSMITH_TRACING=false .venv/bin/python scripts/run_research.py index --technology KIVI
LANGSMITH_TRACING=false .venv/bin/python scripts/run_research.py index --technology InfiniGen

# 설정한 생성 모델과 API 키를 사용한 한 질문 검증; 파일명은 기존 결과와 겹치지 않게 지정
LANGSMITH_TRACING=false .venv/bin/python scripts/run_research.py research --technology KIVI --question-id principle --max-attempts 1 --max-llm-calls 2 --output-name research_result_shared_check.json
LANGSMITH_TRACING=false .venv/bin/python scripts/run_research.py research --technology InfiniGen --question-id principle --max-attempts 1 --max-llm-calls 2 --output-name research_result_shared_check.json

# InfiniGen 고정 개발 질문집 검색 평가만 지원
LANGSMITH_TRACING=false .venv/bin/python scripts/run_research.py evaluate --technology InfiniGen --top-k 5
```

임베딩 모델이 이미 캐시된 환경에서만 명령 앞에 `HF_HUB_OFFLINE=1`을 붙여 오프라인으로 실행할 수 있습니다. 기존 `scripts/run_infinigen.py` 명령도 같은 CLI로 연결됩니다. 생성 LLM은 `.env` 또는 셸의 `LLM_PROVIDER`, `LLM_MODEL`, API 키로 지정합니다. 현재 구현된 제공자는 OpenAI이며 Ollama는 사용하지 않습니다. 현재 셸의 `OPENAI_API_KEY`가 `.env`와 다를 때만 앞에 `env -u OPENAI_API_KEY`를 붙여 `.env` 값을 사용하세요. API 키 값은 출력하지 않습니다.

질문은 문제·원리·구현·실험 조건·성능·품질·한계·독립 평가·공개 상태·논문과 공개 구현 연결의 공통 10개입니다. 논문 내용은 RAG, 현재 공개 정보는 Tavily 웹 본문, 연결 질문은 둘 다 검색합니다. 경로 선택은 사실 검증이 아니며, PDF/웹 모두 실제 발췌와 문서·페이지 또는 URL을 확인해야 근거로 남깁니다. 비교 논문의 자체 결과를 대상 기술에 귀속하지 않습니다. 확인되지 않은 항목은 미확인으로 남깁니다.

기존 InfiniGen 고정 개발 질문집 12개의 LangChain 색인 검색 평가는 HitRate@5 0.75(9/12), MRR@5 0.576이며 `data/cache/infinigen/retrieval_eval_langchain.json`에 보존되어 있습니다. 이 수치는 페이지 검색에 관한 것으로 주장 정확도나 다른 환경의 성능을 뜻하지 않습니다. 이전 원시 조사 결과도 `data/cache/infinigen/`에 보존합니다.

## State와 노드 규약

`State`는 전체 실행의 공유 데이터입니다. `new_state()`에서 입력과 미생성 결과를 초기화합니다. 결과 `None`은 미생성, `evidence_gaps=None`은 미검토, `evidence_gaps=[]`는 검토 후 공백 없음입니다. `score=None`은 미판단이며 0점과 다릅니다. 노드는 `node(state) -> StateUpdate` 함수로 만들고 **자신이 생산하는 top-level 키만 반환**합니다. 병렬 노드가 전체 State를 돌려주면 충돌합니다. 누적 공유 키와 reducer는 현재 필요하지 않습니다.

| State 키 | 생산자 | 주 소비자 |
| --- | --- | --- |
| `selected_technologies`, `domain_and_criteria`, `max_research_rounds` | `new_state`/입력 | 모든 조사·평가, 입력 확인·라우팅 |
| `kivi_evidence` | KIVI 조사 | 네 평가, 근거 확인, 보고서 |
| `infinigen_evidence` | InfiniGen 조사 | 네 평가, 근거 확인, 보고서 |
| `maturity_eval` | TRL 평가 | 근거 확인, 종합 |
| `market_eval` | 시장성 평가 | 근거 확인, 종합 |
| `stakeholder_eval` | 이해관계자 평가 | 근거 확인, 종합 |
| `domain_eval` | 도메인 평가 | 근거 확인, 종합 |
| `evidence_gaps` | 근거 확인 | 재조사, 종합, 보고서 |
| `research_round` | 초기값/재조사 횟수 노드 | 재조사 라우팅 |
| `synthesis` | 종합 | 보고서 |
| `report` | 보고서 | 최종 출력 |

근거는 ID, 기술, 주장/발췌, 문서·URL·페이지, 모델·워크로드·비교 기준, 한계, 출처 확인 상태를 담습니다. 평가는 기술·기준·판단·nullable 점수·이유·근거 ID·불확실성·근거 상태를 연결합니다. TRL은 공개 정보에 근거한 추정이며 미공개 정보가 낮은 TRL을 뜻하지 않습니다. 이해관계자 항목은 직접 확인한 반응(`source_checked`)과 근거 기반 추론(`inferred`), 미확인(`unverified`)을 구별합니다. `public_estimate`는 공개 자료 기반 TRL 추정에 사용합니다. PDF 전문, 모델·벡터 DB 객체, 비밀정보는 State에 넣지 않습니다.

## 기능별 시작 파일

| 담당 기능 | 시작 파일 |
| --- | --- |
| 공통 계약·초기값 | `src/kv_cache_eval/common/schemas.py`, `state.py` |
| 기술 조사·RAG | `src/kv_cache_eval/features/technical_research/node.py`, `ingest.py`, `retriever.py`, `workflow.py`, `sources/` |
| TRL | `src/kv_cache_eval/features/maturity/node.py` |
| 시장성 | `src/kv_cache_eval/features/market/node.py` |
| 이해관계자 | `src/kv_cache_eval/features/stakeholders/node.py` |
| 도메인 | `src/kv_cache_eval/features/domain/node.py` |
| 종합 | `src/kv_cache_eval/features/synthesis/node.py` |
| 보고서 | `src/kv_cache_eval/features/report/node.py` |
| 배선·재조사 | `src/kv_cache_eval/graph/workflow.py`, `gates.py` |

KIVI와 InfiniGen 결과는 각각 별도 State 키에 반환합니다. 고른 RAG 원문 총합은 **200페이지 이하**로 관리합니다. 200페이지는 채울 목표나 보고서 분량이 아닙니다. [모델 카드](https://huggingface.co/BAAI/bge-m3)는 검색 질의에 별도의 instruction 접두어를 요구하지 않습니다. 원문·인덱스·캐시·`.env`는 저장소에 포함하지 않습니다.

보고서는 **SUMMARY**로 시작해 **REFERENCE**로 끝나며 실제 사용한 근거만 인용해야 합니다. PDF 출력은 후속 기능입니다.

## 실습 참고

읽기 전용 참고 경로: `/Users/bmh7190/skala/skala-rag/langgraph/`의 `00-Basic/02-State.ipynb`, `00-Basic/03-Graph.ipynb`, `01-Features/21-Branching.ipynb`, `10-Agent/11-Multi-ReportAgent.ipynb`, `20-RAG/13-AgenticRAG.ipynb`와 `20-RAG/rag/{base,pdf,utils}.py`. 이 프로젝트는 실습의 함수형 `TypedDict`/`StateGraph` 패턴만 가져오며, 실습 노트북·출력·프롬프트·API 키를 복사하지 않습니다. 실습의 큰 1.x 의존성 목록 대신 현재 뼈대에 필요한 `langgraph`만 필수로 선언하고 후속 개발 패키지는 옵션으로 분리했습니다.
