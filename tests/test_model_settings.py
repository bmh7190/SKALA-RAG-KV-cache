"""모델 설정 우선순위와 평가 내 설정 고정을 외부 호출 없이 검증한다."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dotenv import dotenv_values

from kv_cache_eval.common.config import model_settings
from kv_cache_eval.common.llm import chat_model
from kv_cache_eval.features.domain import runtime

FILE_VALUES = {
    "LLM_PROVIDER": "openai",
    "LLM_MODEL": "file-model",
    "OPENAI_API_KEY": "file-test-key",
}


class ModelSettingsTest(unittest.TestCase):
    def test_process_overrides_file_without_mutating_environment(self):
        environment = {
            "LLM_PROVIDER": " OPENAI ",
            "LLM_MODEL": " shell-model ",
            "OPENAI_API_KEY": " shell-test-key ",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertLogs(
                "kv_cache_eval.common.config", level="WARNING"
            ) as logs:
                settings = model_settings(file_values=FILE_VALUES)
            self.assertEqual(dict(os.environ), environment)
        self.assertEqual(settings["model"], "shell-model")
        self.assertEqual(settings["api_key"], "shell-test-key")
        self.assertEqual(settings["max_retries"], 0)
        self.assertNotIn("shell-test-key", " ".join(logs.output))
        self.assertNotIn("file-test-key", " ".join(logs.output))

    def test_provider_alias_uses_process_layer_and_rejects_conflicts(self):
        aliases = ("LLM_PROVIDER", "LM_PROVIDER")
        for name in aliases:
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: "openai"}, clear=True),
            ):
                settings = model_settings(
                    file_values={**FILE_VALUES, "LLM_PROVIDER": "other"},
                    provider_names=aliases,
                )
                self.assertEqual(settings["model"], "file-model")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "서로 다릅니다"):
                model_settings(
                    file_values={**FILE_VALUES, "LM_PROVIDER": "other"},
                    provider_names=aliases,
                )

    def test_explicit_empty_credentials_do_not_fall_back_to_file(self):
        for name in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY"):
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: " "}, clear=True),
            ):
                with self.assertRaises(ValueError):
                    model_settings(file_values=FILE_VALUES)

    def test_factory_selects_judge_and_preserves_timeouts(self):
        with (
            patch.dict(
                os.environ, {**FILE_VALUES, "JUDGE_MODEL": " judge-model "}, clear=True
            ),
            patch("kv_cache_eval.common.llm.load_environment"),
            patch("langchain_openai.ChatOpenAI") as factory,
        ):
            chat_model(role="judge", timeout=45)
            self.assertEqual(
                factory.call_args.kwargs,
                {
                    "model": "judge-model",
                    "api_key": "file-test-key",
                    "timeout": 45,
                    "max_retries": 0,
                },
            )
            os.environ["JUDGE_MODEL"] = ""
            chat_model(role="judge")
            self.assertEqual(factory.call_args.kwargs["model"], "file-model")
            self.assertEqual(factory.call_args.kwargs["timeout"], 120)

    def test_explicit_settings_skip_environment_loading(self):
        settings = {
            "model": "snapshot",
            "api_key": "test-key",
            "timeout": 60,
            "max_retries": 0,
        }
        with (
            patch("kv_cache_eval.common.llm.load_environment") as load,
            patch("langchain_openai.ChatOpenAI") as factory,
        ):
            chat_model(settings=settings)
        load.assert_not_called()
        factory.assert_called_once_with(**settings)

    def test_domain_file_alias_and_timeout_do_not_leak_to_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "domain.env"
            env_file.write_text(
                "LM_PROVIDER=openai\nLLM_MODEL=domain-model\n"
                "OPENAI_API_KEY=domain-test-key\nDOMAIN_LLM_TIMEOUT_SECONDS=17.5\n"
            )
            with patch.dict(os.environ, {"DOMAIN_ENV_FILE": str(env_file)}, clear=True):
                settings = runtime._load_settings(dotenv_values)
                self.assertEqual(dict(os.environ), {"DOMAIN_ENV_FILE": str(env_file)})
                self.assertEqual(settings["model"], "domain-model")
                self.assertEqual(settings["timeout"], 17.5)
                os.environ["DOMAIN_LLM_TIMEOUT_SECONDS"] = "25"
                self.assertEqual(runtime._load_settings(dotenv_values)["timeout"], 25)
                env_file.unlink()
                with self.assertRaisesRegex(
                    runtime.DomainConfigurationError, "파일이 없습니다"
                ):
                    runtime._load_settings(dotenv_values)

    def test_domain_timeout_defaults_and_invalid_values(self):
        with (
            patch.dict(os.environ, FILE_VALUES, clear=True),
            patch.object(runtime, "find_env_file", return_value=None),
        ):
            self.assertEqual(runtime._load_settings(dotenv_values)["timeout"], 60)
            for invalid in ("nan", "inf", "0", "-1", "abc"):
                with self.subTest(invalid=invalid):
                    os.environ["DOMAIN_LLM_TIMEOUT_SECONDS"] = invalid
                    with self.assertRaisesRegex(
                        runtime.DomainConfigurationError, "양의 유한한"
                    ):
                        runtime._load_settings(dotenv_values)

    def test_domain_configuration_errors_keep_domain_error_type(self):
        with (
            patch.dict(os.environ, {**FILE_VALUES, "OPENAI_API_KEY": ""}, clear=True),
            patch.object(runtime, "find_env_file", return_value=None),
        ):
            with self.assertRaises(runtime.DomainConfigurationError):
                runtime._load_settings(dotenv_values)

    def test_domain_freezes_settings_until_next_evaluation(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = "result"
        with (
            patch.dict(os.environ, FILE_VALUES, clear=True),
            patch.object(runtime, "find_env_file", return_value=None),
            patch("kv_cache_eval.common.llm.chat_model", return_value=model) as factory,
        ):
            with runtime.evaluation_settings():
                self.assertEqual(runtime.invoke_structured([], dict), "result")
                os.environ["LLM_MODEL"] = "next-model"
                runtime.invoke_structured([], dict)
            with runtime.evaluation_settings():
                runtime.invoke_structured([], dict)
        self.assertEqual(
            [call.kwargs["settings"]["model"] for call in factory.call_args_list],
            ["file-model", "file-model", "next-model"],
        )
        model.with_structured_output.assert_called_with(
            dict, method="json_schema", strict=True
        )


if __name__ == "__main__":
    unittest.main()
