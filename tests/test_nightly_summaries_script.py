from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "nightly_summaries.sh"


class NightlySummariesScriptTests(unittest.TestCase):
    def test_nightly_uses_flash_with_pro_quality_fallback(self):
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('NIGHTLY_SUMMARY_MODEL:-deepseek-v4-flash', script)
        self.assertIn('NIGHTLY_SUMMARY_FALLBACK_MODEL:-deepseek-v4-pro', script)
        self.assertIn('scripts/build_deepseek_summaries.py', script)
        self.assertIn(
            '--model "$SUMMARY_MODEL" --fallback-model "$SUMMARY_FALLBACK_MODEL"', script,
        )

    def test_codex_quota_only_guards_codex_reocr(self):
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('NIGHTLY_REOCR_LLM:-deepseek:deepseek-v4-pro', script)
        self.assertIn('[[ "$REOCR_LLM" == codex_cli:* ]]', script)
        self.assertIn(
            'nightly re-OCR skipped: weekly quota floor or quota unavailable', script,
        )
        summary_call = script.index('scripts/build_deepseek_summaries.py')
        quota_call = script.index('-m src.codex_quota')
        self.assertLess(quota_call, summary_call)


if __name__ == "__main__":
    unittest.main()
