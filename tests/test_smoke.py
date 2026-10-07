"""State isolation and structural evidence checks without network access."""

import unittest

from kv_cache_eval.common.state import new_state
from kv_cache_eval.graph.gates import check_evidence


class SmokeTest(unittest.TestCase):
    def test_fresh_state_and_empty_evidence_are_distinct(self):
        first, second = new_state(), new_state()
        self.assertIsNone(first["evidence_gaps"])
        self.assertNotEqual(first["trace_id"], second["trace_id"])
        first["completed_agents"].append("market")
        self.assertEqual(second["completed_agents"], [])
        first["kivi_evidence"] = {"evidence": [], "notes": []}
        checked = check_evidence(first)
        self.assertTrue(checked["evidence_gaps"])
        self.assertFalse(checked["evidence_decision"]["ready"])
