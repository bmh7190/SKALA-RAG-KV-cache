"""하드코딩한 질문으로 KIVI와 InfiniGen 기술 조사를 순서대로 실행한다."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.technical_research.node import research_technology


QUESTION = (
    "GPU 기반 클라우드 LLM 서비스 환경에서 KV cache 최적화 기술인 SW 진영의 KIVI와 "
    "HW 메모리 접근 진영의 InfiniGen을 조사해 줘. 논문과 공개 웹 자료를 바탕으로 "
    "기술 원리, 성능과 한계, TRL·시장성·이해관계자·도메인 적합성을 평가할 근거와 출처를 정리해 줘."
)
RUNS_DIR = Path("data/cache/runs")


def main() -> int:
    state = new_state()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = RUNS_DIR / f"research-{stamp}-{uuid4().hex[:8]}.json"
    record = {"question": QUESTION, "status": "running", "completed_technologies": [], "state": state}

    def save() -> None:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    print(f"질문 저장: {path}")
    for technology, key in (("KIVI", "kivi_evidence"), ("InfiniGen", "infinigen_evidence")):
        print(f"{technology} 조사 시작")
        try:
            result = research_technology(state, technology, user_question=QUESTION)
        except Exception as error:
            record["status"] = "failed"
            record["failed_technology"] = technology
            record["error_type"] = type(error).__name__
            save()
            print(f"{technology} 조사 실패({type(error).__name__}); 중간 결과: {path}", file=sys.stderr)
            return 1
        state[key] = result
        record["completed_technologies"].append(technology)
        save()
        print(f"{technology} 조사 완료: 근거 {len(result['evidence'])}개; 중간 결과: {path}")
    record["status"] = "completed"
    save()
    print(f"두 기술 조사 결과: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
