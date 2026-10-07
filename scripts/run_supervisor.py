"""Run, pause or resume the same durable Supervisor graph."""
import argparse
import json
from kv_cache_eval.graph import run

DEFAULT_QUESTION = ('GPU 기반 클라우드 LLM 서비스 환경에서 KIVI와 InfiniGen의 원리·성능·한계를 조사하고 '
                    '기술 성숙도·시장성·이해관계자·도메인 적용성을 중립적으로 비교해 줘.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', default=DEFAULT_QUESTION)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pause-after', action='append', default=None)
    parser.add_argument('--checkpoint', default='data/cache/supervisor/checkpoints.sqlite')
    parser.add_argument('--max-steps', type=int, default=30)
    args = parser.parse_args()
    state = run(args.question, run_id=args.run_id, resume=args.resume,
                checkpoint_path=args.checkpoint, interrupt_after=args.pause_after,max_steps=args.max_steps)
    print(json.dumps({key:state[key] for key in ('trace_id','status','step_count','report_revision','termination_reason','pdf_path')},ensure_ascii=False,indent=2))
    return 0 if state['status'] in ('running','completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
