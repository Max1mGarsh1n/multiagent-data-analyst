import json
import re

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return THINK_BLOCK_RE.sub("", text).strip()


def extract_json_object(text: str) -> str:
    """Extract the first valid JSON object from model output, stripping markdown fences."""
    cleaned = strip_thinking(text)
    cleaned = FENCE_RE.sub("", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return cleaned
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(cleaned[index:])
            if isinstance(obj, dict):
                return cleaned[index : index + end]
        except json.JSONDecodeError:
            continue

    raise ValueError("Модель не вернула корректный JSON-объект.")
