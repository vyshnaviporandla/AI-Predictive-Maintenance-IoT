"""Local Ollama integration for evidence-grounded maintenance guidance."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:0.5b"
REQUEST_TIMEOUT_SECONDS = 300


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot return a valid response."""


def ollama_settings():
    """Return configured Ollama URL and model name."""
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    return base_url, model


def generate_maintenance_advice(
    sensor_readings,
    prediction,
    probability,
    alert,
    model=None,
    base_url=None,
):
    """Ask a local Ollama model to explain one model prediction."""
    configured_url, configured_model = ollama_settings()
    base_url = (base_url or configured_url).rstrip("/")
    model = model or configured_model

    if not base_url.startswith(("http://", "https://")):
        raise OllamaError("OLLAMA_BASE_URL must start with http:// or https://.")
    if not model.strip():
        raise OllamaError("Choose a non-empty Ollama model name.")

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cautious industrial maintenance assistant. "
                    "Explain the provided classifier result using only the supplied "
                    "sensor readings and model output. Distinguish measured facts "
                    "from hypotheses. Do not claim certainty, invent a failure mode, "
                    "or override the classifier. Give concise next inspection steps. "
                    "State that a qualified technician must validate recommendations "
                    "and that this synthetic-data prototype is not a safety system."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Analyze this simulated equipment reading.\n"
                    f"Sensor readings: {json.dumps(sensor_readings, sort_keys=True)}\n"
                    f"Classifier prediction: {int(prediction)} "
                    f"({'failure risk' if prediction else 'no failure predicted'})\n"
                    f"Classifier failure probability: {float(probability):.3f}\n"
                    f"Existing rule-based alert: {alert}\n\n"
                    "Return: (1) a short explanation, (2) the readings worth checking "
                    "first, and (3) practical inspection actions. Keep it under 180 words."
                ),
            },
        ],
        "options": {"temperature": 0.2},
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if exc.code == 404:
            raise OllamaError(
                f"Ollama could not find model '{model}'. Pull it with "
                f"`ollama pull {model}` and try again."
            ) from exc
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail or exc.reason}") from exc
    except TimeoutError as exc:
        raise OllamaError(
            f"Ollama did not finish generating a response within "
            f"{REQUEST_TIMEOUT_SECONDS} seconds. Try a smaller prompt or a faster machine."
        ) from exc
    except URLError as exc:
        raise OllamaError(
            "Could not connect to Ollama. Start the local Ollama service and confirm "
            f"it is reachable at {base_url}."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OllamaError("Ollama returned an invalid JSON response.") from exc

    try:
        advice = result["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError) as exc:
        raise OllamaError("Ollama's response did not contain assistant message text.") from exc
    if not advice:
        raise OllamaError("Ollama returned an empty maintenance explanation.")
    return advice
