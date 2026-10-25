import json
import time

import httpx

import config

CHAT_URL = f"{config.OLLAMA_URL}/api/chat"
MODEL = config.CODE_MODEL  # kept for backwards compatibility


class OllamaError(Exception):
    """A problem talking to Ollama, with a message that is safe to show the user."""


def _explain(status, body, model):
    try:
        detail = json.loads(body).get("error", body)
    except (ValueError, AttributeError):
        detail = body
    if status == 404 or "not found" in str(detail).lower():
        return f"Model '{model}' is not installed. Install it with:  ollama pull {model}"
    return f"Ollama returned an error ({status}): {detail}"


def _options(temperature, num_predict):
    return {
        "num_ctx": config.NUM_CTX,
        "num_predict": num_predict or config.NUM_PREDICT,
        "temperature": temperature,
    }


async def stream_chat(messages, model, temperature=0.3, num_predict=None):
    """Yield {"text": str} pieces, then one {"done": True, "stats": {...}}."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": config.KEEP_ALIVE,
        "options": _options(temperature, num_predict),
    }
    timeout = httpx.Timeout(connect=5, read=600, write=30, pool=5)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", CHAT_URL, json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise OllamaError(_explain(response.status_code, body, model))
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaError(str(data["error"]))
                    piece = (data.get("message") or {}).get("content", "")
                    if piece:
                        yield {"text": piece}
                    if data.get("done"):
                        count = data.get("eval_count") or 0
                        duration = (data.get("eval_duration") or 0) / 1e9
                        yield {
                            "done": True,
                            "stats": {
                                "tokens": count,
                                "tokens_per_second": round(count / duration, 1) if duration else None,
                                "prompt_tokens": data.get("prompt_eval_count"),
                            },
                        }
                        return
    except httpx.ConnectError as error:
        raise OllamaError(
            f"Could not connect to Ollama at {config.OLLAMA_URL}. "
            "Start the Ollama app (or run `ollama serve`) and try again."
        ) from error
    except httpx.ReadTimeout as error:
        raise OllamaError("Ollama stopped responding. The model may be overloaded.") from error


async def chat_with_ollama(messages, tools=None, model=None):
    """Non-streaming call (original helper, kept for compatibility)."""
    text = ""
    async for chunk in stream_chat(messages, model or config.CODE_MODEL):
        text += chunk.get("text", "")
    return {"message": {"role": "assistant", "content": text}}


_cache = {"at": 0.0, "models": []}


async def list_models(force=False):
    """Names of installed models (cached for a few seconds)."""
    if not force and time.time() - _cache["at"] < 5:
        return _cache["models"]
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{config.OLLAMA_URL}/api/tags")
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise OllamaError(
            f"Could not connect to Ollama at {config.OLLAMA_URL}. "
            "Start the Ollama app (or run `ollama serve`) and try again."
        ) from error
    models = [m["name"] for m in response.json().get("models", [])]
    _cache.update(at=time.time(), models=models)
    return models


async def loaded_models():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{config.OLLAMA_URL}/api/ps")
            return [m["name"] for m in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def _has(installed, name):
    if name in installed:
        return name
    if ":" not in name and f"{name}:latest" in installed:
        return f"{name}:latest"
    return None


async def resolve_model(preferred, alternates=()):
    """Pick an installed model. Returns (model_name, warning_or_None)."""
    installed = await list_models()
    found = _has(installed, preferred)
    if found:
        return found, None
    for alt in alternates:
        found = _has(installed, alt)
        if found:
            return found, (
                f"Model '{preferred}' is not installed, so '{found}' was used instead. "
                f"For better results run:  ollama pull {preferred}"
            )
    usable = [m for m in installed if "embed" not in m.lower()]
    if usable:
        return usable[0], (
            f"Model '{preferred}' is not installed, so '{usable[0]}' was used instead. "
            f"Install it with:  ollama pull {preferred}"
        )
    raise OllamaError(f"No models are installed. Run:  ollama pull {preferred}")
