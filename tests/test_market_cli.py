"""시장성 단독 실행 명령 테스트."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from kv_cache_eval.features.market.cli import main as market_cli


class MarketCliTest(unittest.TestCase):
    def test_checks_configuration_without_api_calls(self):
        environment = {
            "TAVILY_API_KEY": "tavily-test",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "test-model",
            "OPENAI_API_KEY": "openai-test",
        }
        output = StringIO()
        with patch.dict("os.environ", environment, clear=True), redirect_stdout(output):
            exit_code = market_cli(["--check-config"])
        self.assertEqual(exit_code, 0)
        self.assertIn("provider=openai, model=test-model", output.getvalue())


if __name__ == "__main__":
    unittest.main()
