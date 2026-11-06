"""Keep what we send to the model inside its context window.

Token counts are estimates (characters / CHARS_PER_TOKEN). The estimate is corrected per
model from what Ollama actually reports (record_actual), and every message is charged a few
tokens for its role markers, so the budget stays honest for the model really in use.
"""
import threading

import config

PER_MESSAGE_OVERHEAD = 8          # role markers / template tokens around each message
SYSTEM_SHARE_FALLBACK = 0.55      # most of the budget the system prompt may take
_CALIBRATION = {}                 # model name -> correction factor
_CALIBRATION_LOCK = threading.Lock()
MIN_FACTOR, MAX_FACTOR = 0.85, 2.5


def calibration_factor(model):
    with _CALIBRATION_LOCK:
        return _CALIBRATION.get(model, 1.0)


def record_actual(model, raw_estimate, actual):
    """Learn from a real prompt-token count (Ollama's prompt_eval_count).

    A count far below the estimate usually means Ollama reused a cached prompt prefix and only
    counted the new part, so those are ignored rather than trusted.
    """
    if not model or raw_estimate <= 0 or not actual or actual <= 0:
        return
    ratio = actual / raw_estimate
    if ratio < 0.7:
        return
    ratio = max(MIN_FACTOR, min(MAX_FACTOR, ratio))
    with _CALIBRATION_LOCK:
        old = _CALIBRATION.get(model)
        _CALIBRATION[model] = ratio if old is None else max(
            MIN_FACTOR, min(MAX_FACTOR, 0.6 * old + 0.4 * ratio))


def reset_calibration():
    with _CALIBRATION_LOCK:
        _CALIBRATION.clear()


def estimate_tokens(text: str, model=None) -> int:
    base = int(len(text) / config.CHARS_PER_TOKEN) + 4
    if model:
        return int(base * calibration_factor(model)) + 1
    return base


def _total(messages, model=None):
    return sum(estimate_tokens(m["content"], model) + PER_MESSAGE_OVERHEAD for m in messages)


def prompt_tokens(messages, model=None):
    """Estimated size of a whole prompt, including per-message overhead."""
    return _total(messages, model)


DIGEST_MARKER = "EARLIER IN THIS CONVERSATION"


def digest_chars(budget=None):
    """How much room the dropped-history digest may take: a small share of the window."""
    budget = budget or config.PROMPT_BUDGET_TOKENS
    return max(400, min(1400, int(budget * config.CHARS_PER_TOKEN * 0.07)))


def _one_line(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def make_digest(dropped, limit=1400):
    """A compact, mechanical reminder of turns that no longer fit. No model is involved.

    It is made from the real messages (first words of each question and answer), newest
    first so the most recent context survives the character cap.
    """
    lines = []
    for message in reversed(dropped):
        if message["role"] == "user" and message["content"].startswith("TOOL RESULT"):
            continue
        who = "You asked" if message["role"] == "user" else "Atlas replied"
        lines.append(f"- {who}: {_one_line(message['content'], 170)}")

    kept, used = [], 0
    for line in lines:
        if used + len(line) > limit:
            break
        kept.append(line)
        used += len(line) + 1

    if not kept:
        return ""

    kept.reverse()
    omitted = len(lines) - len(kept)
    more = f" ({omitted} older lines not shown)" if omitted > 0 else ""
    return (
        f"\n\n{DIGEST_MARKER} (compressed automatically; may be incomplete or wrong, "
        f"so re-check the source before relying on it){more}:\n" + "\n".join(kept)
    )


def _middle_cut(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        text[:head]
        + f"\n\n[... {omitted} characters omitted because they do not fit in the model's "
        "context window ...]\n\n"
        + text[-tail:]
    )


PROJECT_MARKER = "\n\nACTIVE PROJECT (id:"


def _shrink_system(content, max_chars):
    """Cut an oversized system prompt where it is cheapest: the project overview, not the rules."""
    index = content.find(PROJECT_MARKER)
    if 0 < index < max_chars - 400:
        head, project = content[:index], content[index:]
        keep = max_chars - len(head) - 80
        return head + project[:keep] + "\n[... project overview shortened to fit the model's memory ...]"
    return _middle_cut(content, max_chars)


def _shorten_tool_result(message):
    message["content"] = (
        message["content"][:300] + "\n[older tool result shortened to save space]"
    )


def fit_messages(messages, budget=None, model=None, protect_from=None):
    """Return (messages, notes). Messages[0] is the system prompt, the last is the new request.

    Order of sacrifice: an oversized system prompt loses part of its project overview, older
    tool results are shortened, then the oldest chat history is dropped, and only as a last
    resort the newest message is cut in the middle.

    `protect_from` is the index of the message that starts the current turn. Everything from
    there on (the request and the evidence gathered for it) is kept whole while older history
    and older turns' tool results are given up first; this turn's own tool results are only
    shortened once nothing older is left to give up.
    """
    budget = budget or config.PROMPT_BUDGET_TOKENS
    messages = [dict(m) for m in messages]
    notes = []
    protected = len(messages) if protect_from is None else max(1, min(protect_from, len(messages)))

    # 0. The system prompt may not crowd out the conversation.
    share = getattr(config, "SYSTEM_SHARE", SYSTEM_SHARE_FALLBACK)
    system_cap_chars = int(budget * config.CHARS_PER_TOKEN * (share + 0.1))
    if messages and len(messages[0]["content"]) > system_cap_chars > 800:
        messages[0]["content"] = _shrink_system(messages[0]["content"], system_cap_chars)
        notes.append("The project overview was shortened to fit the model's memory.")

    if _total(messages, model) <= budget:
        return messages, notes

    def is_tool_result(message):
        return message["role"] == "user" and message["content"].startswith("TOOL RESULT")

    # 1. Shorten tool results from before the current turn (never the final message).
    for index in range(1, min(protected, len(messages) - 1)):
        if _total(messages, model) <= budget:
            break
        if is_tool_result(messages[index]):
            _shorten_tool_result(messages[index])

    # 2. Drop the oldest history, leaving a short fallible digest of what was dropped.
    dropped = []
    limit = digest_chars(budget)
    room = budget - estimate_tokens("x" * limit, model)
    while _total(messages, model) > room and len(messages) > 2 and protected > 1:
        dropped.append(messages.pop(1))
        protected -= 1
        if "Older messages were left out to fit the model's memory." not in notes:
            notes.append("Older messages were left out to fit the model's memory.")

    if dropped and DIGEST_MARKER not in messages[0]["content"]:
        messages[0]["content"] += make_digest(dropped, limit)

    # 2b. Only now shorten this turn's older tool results, oldest first.
    if protect_from is not None:
        for index in range(1, len(messages) - 1):
            if _total(messages, model) <= budget:
                break
            if is_tool_result(messages[index]) and "[older tool result shortened" not in messages[index]["content"]:
                _shorten_tool_result(messages[index])
    else:
        # without a protected turn the original behaviour applies: drop until it fits
        while _total(messages, model) > room and len(messages) > 2:
            dropped.append(messages.pop(1))
            if "Older messages were left out to fit the model's memory." not in notes:
                notes.append("Older messages were left out to fit the model's memory.")

    # 3. Cut the newest message if it alone is still too big.
    if _total(messages, model) > budget:
        system_tokens = estimate_tokens(messages[0]["content"], model)
        allowed = max(500, budget - system_tokens - 200)
        allowed_chars = int(allowed * config.CHARS_PER_TOKEN)
        if len(messages[-1]["content"]) > allowed_chars:
            messages[-1]["content"] = _middle_cut(messages[-1]["content"], allowed_chars)
            notes.append(
                "Your message is longer than the model can read at once (about "
                f"{config.NUM_CTX} tokens), so the middle was cut. Upload big code as a "
                "ZIP or attach files so the assistant can read them in pieces."
            )

    return messages, notes
