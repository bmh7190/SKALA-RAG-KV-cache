"""전체 그래프와 별개로 시장성 노드만 실행하는 명령줄 진입점."""

import argparse
import json
import os
from collections.abc import Sequence

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.market.node import evaluate, validate_runtime_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kv-market-eval",
        description="KIVI와 InfiniGen의 시장성 웹 조사 및 평가를 실행합니다.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="API를 호출하지 않고 필요한 환경변수만 확인합니다.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="평가 결과 JSON을 들여쓰기 없이 출력합니다.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_environment()
    config = validate_runtime_config(os.environ)
    if args.check_config:
        print(f"시장성 실행 설정 확인 완료: provider={config.provider}, model={config.model}")
        return 0

    update = evaluate(new_state())
    print(json.dumps(update, ensure_ascii=False, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
