from agent.router import detect_mode


def test_plain_chat():
    assert detect_mode("Summarise this market list: rice 5000, beans 3000", False, None) == "chat"
    assert detect_mode("write a polite email asking for a refund", False, None) == "chat"


def test_code_signals():
    assert detect_mode("why does this throw a NullPointerException in my java method", False, None) == "code"
    assert detect_mode("fix this\n```py\nprint(1)\n```", False, None) == "code"
    assert detect_mode("hello", True, None) == "code"          # project attached


def test_short_followup_keeps_mode_long_message_does_not():
    assert detect_mode("make it shorter", False, "code") == "code"
    assert detect_mode("x " * 200, False, "code") == "chat"
