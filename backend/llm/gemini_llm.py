import time

import requests

from backend.config import get_gemini_api_key, get_gemini_model


class GeminiAPIError(RuntimeError):
    """Raised when Gemini cannot generate a usable text response."""


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


def generate_text(
    prompt: str,
    max_output_tokens: int = 4096,
    timeout: int = 120,
) -> str:
    """Generate text through the Gemini generateContent REST API."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The Gemini prompt cannot be empty.")

    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive.")

    model = get_gemini_model()
    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {
                "thinkingLevel": "minimal",
            },
        },
    }

    headers = {
        "x-goog-api-key": get_gemini_api_key(),
        "Content-Type": "application/json",
    }

    response = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise GeminiAPIError(
                    f"Could not connect to Gemini after "
                    f"{MAX_ATTEMPTS} attempts: {exc}"
                ) from exc

            time.sleep(2**attempt)
            continue

        if (
            response.status_code not in RETRYABLE_STATUS_CODES
            or attempt == MAX_ATTEMPTS - 1
        ):
            break

        time.sleep(2**attempt)

    if response is None:
        raise GeminiAPIError("Gemini did not return an HTTP response.")

    try:
        data = response.json()
    except ValueError as exc:
        raise GeminiAPIError(
            f"Gemini returned a non-JSON response " f"(HTTP {response.status_code})."
        ) from exc

    if not response.ok:
        error = data.get("error", {})
        message = error.get("message", "Unknown Gemini API error")

        raise GeminiAPIError(
            f"Gemini API returned HTTP {response.status_code}: {message}"
        )

    candidates = data.get("candidates", [])

    if not candidates:
        raise GeminiAPIError(
            "Gemini returned no answer. The request may have been blocked."
        )

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "")

    if finish_reason == "MAX_TOKENS":
        raise GeminiAPIError(
            "Gemini stopped because the output token limit was reached. "
            "Increase max_output_tokens or reduce the thinking level."
        )

    parts = candidate.get("content", {}).get("parts", [])

    answer = "".join(
        part.get("text", "") for part in parts if isinstance(part.get("text"), str)
    ).strip()

    if not answer:
        raise GeminiAPIError(
            f"Gemini returned no text. Finish reason: " f"{finish_reason or 'unknown'}"
        )

    return answer
