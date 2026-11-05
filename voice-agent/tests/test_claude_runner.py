import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from claude_runner import ClaudeConfig, build_command, clean_for_speech, parse_output  # noqa: E402


class BuildCommandTests(unittest.TestCase):
    def test_first_turn_has_no_resume(self):
        cmd = build_command(ClaudeConfig(), "cześć", None)
        self.assertEqual(cmd[:3], ["claude", "-p", "cześć"])
        self.assertIn("--output-format", cmd)
        self.assertNotIn("--resume", cmd)

    def test_next_turn_resumes_session(self):
        cmd = build_command(ClaudeConfig(), "dalej", "abc-123")
        self.assertEqual(cmd[cmd.index("--resume") + 1], "abc-123")

    def test_optional_flags(self):
        cfg = ClaudeConfig(model="sonnet", permission_mode="acceptEdits",
                           allowed_tools=["Read", "Bash(git status)"])
        cmd = build_command(cfg, "x", None)
        self.assertEqual(cmd[cmd.index("--model") + 1], "sonnet")
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "acceptEdits")
        i = cmd.index("--allowedTools")
        self.assertEqual(cmd[i + 1:i + 3], ["Read", "Bash(git status)"])


class ParseOutputTests(unittest.TestCase):
    def test_real_shape(self):
        out = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                          "result": "ok", "session_id": "3f43"})
        reply = parse_output(out)
        self.assertEqual(reply.text, "ok")
        self.assertEqual(reply.session_id, "3f43")
        self.assertFalse(reply.is_error)

    def test_json_after_noise(self):
        out = "jakiś log\n" + json.dumps({"result": "tak", "session_id": "s1"})
        self.assertEqual(parse_output(out).text, "tak")

    def test_garbage_is_error(self):
        reply = parse_output("nie json")
        self.assertTrue(reply.is_error)
        self.assertEqual(reply.text, "nie json")

    def test_empty_result_falls_back(self):
        reply = parse_output(json.dumps({"result": "", "subtype": "error_max_turns", "is_error": True}))
        self.assertEqual(reply.text, "error_max_turns")
        self.assertTrue(reply.is_error)


class CleanForSpeechTests(unittest.TestCase):
    def test_strips_markdown(self):
        text = "# Tytuł\n\n- **Punkt** pierwszy\n- `kod` tu\n\n```py\nprint(1)\n```\nZobacz [docs](https://x.pl)."
        cleaned = clean_for_speech(text)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertNotIn("print", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertIn("Punkt pierwszy", cleaned)
        self.assertIn("Zobacz docs.", cleaned)

    def test_plain_text_untouched(self):
        self.assertEqual(clean_for_speech("Dzień dobry, jak mogę pomóc?"), "Dzień dobry, jak mogę pomóc?")


if __name__ == "__main__":
    unittest.main()
