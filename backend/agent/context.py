"""Keep what we send to the model inside its context window."""
import config


def estimate_tokens(text: str) -> int:
    return int(len(text) / config.CHARS_PER_TOKEN) + 4


def _total(messages):
    return sum(estimate_tokens(m["content"]) for m in messages)


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


def fit_messages(messages, budget=None):
    """Return (messages, notes). Messages[0] is the system prompt, the last is the new request.

    Order of sacrifice: older tool results are shortened, then the oldest chat history is
    dropped, and only as a last resort the newest message is cut in the middle.
    """
    budget = budget or config.PROMPT_BUDGET_TOKENS
    messages = [dict(m) for m in messages]
    notes = []

    if _total(messages) <= budget:
        return messages, notes

    # 1. Shorten old tool results (everything except the final message).
    for message in messages[1:-1]:
        if _total(messages) <= budget:
            break
        if message["role"] == "user" and message["content"].startswith("TOOL RESULT"):
            message["content"] = (
                message["content"][:300]
                + "\n[older tool result shortened to save space]"
            )

    # 2. Drop the oldest history.
    while _total(messages) > budget and len(messages) > 2:
        del messages[1]
        if "Older messages were left out to fit the model's memory." not in notes:
            notes.append("Older messages were left out to fit the model's memory.")

    # 3. Cut the newest message if it alone is still too big.
    if _total(messages) > budget:
        system_tokens = estimate_tokens(messages[0]["content"])
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
