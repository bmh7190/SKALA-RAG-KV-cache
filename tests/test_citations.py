"""원본·도메인 재사용 근거의 ID 충돌 판정을 검증한다."""

from copy import deepcopy
import unittest

from kv_cache_eval.common.citations import collect_verified_evidence
from kv_cache_eval.common.state import new_state


def _evidence():
    return {
        "id": "kivi:paper:performance_results:fixture",
        "technology": "KIVI",
        "claim": "검증용 주장",
        "excerpt": "검증용 발췌",
        "source": {"document": "paper.pdf", "url": None, "page": 2},
        "experiment": {"model": "Llama-2-7B", "workload": "generation tasks"},
        "limitations": ["검증용 한계"],
        "verification_status": "source_checked",
    }


class CitationCollectionTest(unittest.TestCase):
    def _state(self, copied):
        original = _evidence()
        state = new_state()
        state["kivi_evidence"] = {"evidence": [original], "notes": []}
        state["domain_evidence"] = {"evidence": [copied], "notes": []}
        return state, original

    def test_optional_experiment_missing_and_null_are_equivalent(self):
        copied = deepcopy(_evidence())
        copied["experiment"]["baseline"] = None
        state, original = self._state(copied)
        result = collect_verified_evidence(state)
        self.assertIs(result[original["id"]], original)
        self.assertNotIn("baseline", original["experiment"])
        self.assertIn("baseline", copied["experiment"])

    def test_real_same_id_conflicts_still_fail(self):
        variants = {
            "claim": lambda item: item.update(claim="다른 주장"),
            "excerpt": lambda item: item.update(excerpt="다른 발췌"),
            "source": lambda item: item["source"].update(page=3),
            "technology": lambda item: item.update(technology="InfiniGen"),
            "experiment": lambda item: item["experiment"].update(baseline="다른 기준"),
            "limitations": lambda item: item["limitations"].append("추가 한계"),
        }
        for name, change in variants.items():
            with self.subTest(name=name):
                copied = deepcopy(_evidence())
                copied["experiment"]["baseline"] = None
                change(copied)
                state, _ = self._state(copied)
                with self.assertRaisesRegex(ValueError, "서로 다른 근거가 같은 ID"):
                    collect_verified_evidence(state)


if __name__ == "__main__":
    unittest.main()
