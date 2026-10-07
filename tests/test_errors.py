"""공통 오류 계약과 Supervisor 실행 경계의 의존 방향을 외부 API 없이 확인한다."""

import ast
import unittest
from pathlib import Path

from kv_cache_eval.common.errors import (
    AgentFailure,
    InputBudgetExceeded,
    InvalidRequest,
    ResponseRepairExhausted,
    classify_error,
)
from kv_cache_eval.features.domain.results import DomainResponseError
from kv_cache_eval.features.report.node import ReportTooLong, ReportValidationError

SRC = Path(__file__).resolve().parents[1] / "src" / "kv_cache_eval"


class ErrorContractTests(unittest.TestCase):
    def test_feature_failures_carry_their_own_code(self):
        cases = (
            (InvalidRequest("x"), "invalid_request"),
            (InputBudgetExceeded("x"), "input_budget_exceeded"),
            (ResponseRepairExhausted("x"), "response_repair_exhausted"),
            (DomainResponseError("x"), "response_repair_exhausted"),
            (ReportValidationError("x"), "invalid_result"),
            (ReportTooLong("x"), "report_too_long"),
        )
        for error, code in cases:
            with self.subTest(error=type(error).__name__):
                result = classify_error(error)
                self.assertEqual(result["code"], code)
                self.assertFalse(result["retryable"])
                # 기존 `except ValueError` 처리와 호환된다.
                self.assertIsInstance(error, ValueError)

    def test_only_safe_feature_messages_are_exposed(self):
        self.assertEqual(
            classify_error(ReportTooLong("보고서가 11쪽입니다."))["message"],
            "보고서가 11쪽입니다.",
        )
        self.assertEqual(
            classify_error(ReportValidationError("목차 누락"))["message"], "목차 누락"
        )
        hidden = classify_error(InputBudgetExceeded("request body secret"))
        self.assertNotIn("secret", hidden["message"])
        self.assertEqual(
            hidden["message"], "InputBudgetExceeded: input_budget_exceeded"
        )

    def test_external_errors_are_classified_without_feature_types(self):
        class RateLimitError(Exception):
            status_code = 429

        class APIStatusError(Exception):
            status_code = 503

        class AuthenticationError(Exception):
            pass

        cases = (
            (RateLimitError("x"), "rate_limit", True),
            (APIStatusError("x"), "connection", True),
            (TimeoutError("x"), "timeout", True),
            (AuthenticationError("x"), "authentication", False),
            (ValueError("secret credential"), "invalid_result", False),
            (RuntimeError("x"), "execution_error", False),
        )
        for error, code, retryable in cases:
            with self.subTest(error=type(error).__name__):
                result = classify_error(error)
                self.assertEqual(
                    (result["code"], result["retryable"]), (code, retryable)
                )
                self.assertNotIn("secret", result["message"])

    def test_custom_failure_needs_no_supervisor_change(self):
        class NewFeatureError(AgentFailure):
            code = "new_feature_error"
            retryable = True

        self.assertEqual(
            classify_error(NewFeatureError("x")),
            {
                "code": "new_feature_error",
                "message": "NewFeatureError: new_feature_error",
                "retryable": True,
            },
        )

    def test_execution_boundary_does_not_import_feature_exceptions(self):
        path = SRC / "features" / "supervisor" / "execution.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        feature_modules = {
            m for m in modules if m.startswith("kv_cache_eval.features.")
        }
        self.assertTrue(feature_modules)
        for module in feature_modules:
            with self.subTest(module=module):
                # 실행 경계는 Supervisor 밖의 기능 모듈(domain, report 등)을 몰라야 한다.
                self.assertTrue(module.startswith("kv_cache_eval.features.supervisor."))


if __name__ == "__main__":
    unittest.main()
