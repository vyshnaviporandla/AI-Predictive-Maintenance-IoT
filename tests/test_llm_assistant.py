import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.llm_assistant import OllamaError, generate_maintenance_advice


class GenerateMaintenanceAdviceTests(unittest.TestCase):
    def test_posts_only_the_current_prediction_context_and_returns_advice(self):
        response = BytesIO(
            json.dumps(
                {"message": {"content": "Inspect the elevated vibration first."}}
            ).encode("utf-8")
        )

        with patch("src.llm_assistant.urlopen", return_value=response) as urlopen:
            advice = generate_maintenance_advice(
                {"vibration": 3.8},
                prediction=1,
                probability=0.91,
                alert="High risk",
                model="qwen2.5:3b",
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertIn('"vibration": 3.8', payload["messages"][1]["content"])
        self.assertEqual(advice, "Inspect the elevated vibration first.")

    def test_missing_model_has_an_actionable_error(self):
        error = HTTPError(
            "http://localhost:11434/api/chat",
            404,
            "Not Found",
            {},
            BytesIO(b'{"error":"model not found"}'),
        )

        with patch("src.llm_assistant.urlopen", side_effect=error):
            with self.assertRaisesRegex(OllamaError, "ollama pull qwen2.5:0.5b"):
                generate_maintenance_advice({}, 0, 0.1, "Normal")

    def test_unavailable_service_has_an_actionable_error(self):
        with patch("src.llm_assistant.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(OllamaError, "Start the local Ollama service"):
                generate_maintenance_advice({}, 0, 0.1, "Normal")


if __name__ == "__main__":
    unittest.main()
