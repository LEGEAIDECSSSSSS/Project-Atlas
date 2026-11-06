"""Context accounting: per-model calibration, per-message overhead, oversized prompts."""
import config
from agent import context as ctx
from agent import agent as A
from tests.test_agent_gate import run


def setup_function(_):
    ctx.reset_calibration()


def test_estimate_is_calibrated_per_model_from_real_counts():
    text = "x" * 3500
    base = ctx.estimate_tokens(text)
    assert ctx.estimate_tokens(text, "m1") == int(base * 1.0) + 1
    ctx.record_actual("m1", raw_estimate=1000, actual=1500)
    assert ctx.estimate_tokens(text, "m1") > base * 1.4
    assert ctx.estimate_tokens(text, "m2") == int(base) + 1          # other models are untouched


def test_cached_prompt_counts_are_not_trusted():
    ctx.record_actual("m", raw_estimate=1000, actual=100)         # Ollama counted only the new part
    assert ctx.calibration_factor("m") == 1.0
    ctx.record_actual("m", raw_estimate=0, actual=500)
    ctx.record_actual(None, raw_estimate=10, actual=20)
    ctx.record_actual("m", raw_estimate=10, actual=0)
    assert ctx.calibration_factor("m") == 1.0


def test_calibration_is_clamped_and_smoothed():
    ctx.record_actual("m", 100, 100000)
    assert ctx.calibration_factor("m") == ctx.MAX_FACTOR
    ctx.record_actual("m", 1000, 1000)
    assert ctx.MIN_FACTOR <= ctx.calibration_factor("m") < ctx.MAX_FACTOR


def test_every_message_is_charged_overhead():
    msgs = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
    assert ctx.prompt_tokens(msgs) == sum(ctx.estimate_tokens(m["content"]) for m in msgs) + 2 * ctx.PER_MESSAGE_OVERHEAD


def test_oversized_system_prompt_loses_the_project_overview_not_the_rules():
    rules = "RULES " * 100
    system = rules + "\n\nACTIVE PROJECT (id: p)\nThe user attached this project.\n\n" + "FILE " * 6000
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "hi"}]
    fitted, notes = ctx.fit_messages(msgs, budget=1500)
    assert fitted[0]["content"].startswith(rules) and "ACTIVE PROJECT (id: p)" in fitted[0]["content"]
    assert "overview shortened" in fitted[0]["content"] and len(fitted[0]["content"]) < len(system) / 2
    assert any("project overview" in n for n in notes)
    assert fitted[-1]["content"] == "hi"


def test_a_tool_result_cut_by_trimming_keeps_its_closing_marker():
    start, end = A.UNTRUSTED_START, A.UNTRUSTED_END
    system = {"role": "system", "content": "S"}
    request = {"role": "user", "content": "Q"}
    old = {"role": "user", "content": f"TOOL RESULT\n\n{start}\n" + "x" * 5000 + f"\n{end}"}
    newest = {"role": "user", "content": "latest"}
    fitted, _ = A.pinned_fit([system, request, old, newest], (system, request, newest), budget=300)
    for m in fitted:
        if m["content"].startswith("TOOL RESULT"):
            assert m["content"].count(end) == 1


def test_no_room_to_answer_is_reported_instead_of_forcing_a_tiny_answer(monkeypatch):
    monkeypatch.setattr(config, "NUM_CTX", 200)
    out = run("hello there", ["never reached"], project_id=None)
    assert "no room left to answer" in out.text
    assert out.model_calls == 0


def test_the_answer_budget_never_exceeds_what_is_really_left(monkeypatch):
    seen = {}

    async def fake_stream(messages, model, temperature, num_predict=None):
        seen["num_predict"] = num_predict
        yield {"text": "ok"}
        yield {"done": True, "stats": {"prompt_tokens": 50}}

    monkeypatch.setattr(config, "NUM_CTX", 1400)
    monkeypatch.setattr(config, "NUM_PREDICT", 4000)
    out = run("hello there", [], project_id=None, extra_patches=[(A, "stream_chat", fake_stream)])
    assert out.text == "ok"
    est = ctx.prompt_tokens(out.model.calls and [] or [{"role": "system", "content": ""}])
    assert 160 <= seen["num_predict"] <= 1400 - 64


def test_real_prompt_counts_teach_the_estimator_during_a_run(monkeypatch):
    async def fake_stream(messages, model, temperature, num_predict=None):
        yield {"text": "ok"}
        yield {"done": True, "stats": {"prompt_tokens": 5000}}

    run("hello there", [], project_id=None, extra_patches=[(A, "stream_chat", fake_stream)])
    assert ctx.calibration_factor(config.CHAT_MODEL) > 1.0


def test_prefetched_files_too_big_for_the_window_are_reported_not_half_read(project, monkeypatch):
    big = "\n".join(f"def f{i}():\n    return {i}  # " + "x" * 80 for i in range(120))
    w = project({"a.py": big, "b.py": big, "c.py": big})
    monkeypatch.setattr(config, "PROMPT_BUDGET_TOKENS", 1800)
    out = run("Compare a.py and b.py and c.py", ["never reached"], world=w)
    assert "too large to fit" in out.text and out.model_calls == 0


def test_this_turns_evidence_survives_while_older_history_goes_first():
    start, end = A.UNTRUSTED_START, A.UNTRUSTED_END
    system = {"role": "system", "content": "S"}
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"old chat {i} " + "z" * 400}
               for i in range(20)]
    request = {"role": "user", "content": "THE QUESTION"}
    evidence = [{"role": "assistant", "content": '{"tool": "x"}'},
                {"role": "user", "content": f"TOOL RESULT\n\n{start}\n" + "KEEPME " * 150 + f"\n{end}"}]
    messages = [system, *history, request, *evidence]
    fitted, notes = A.pinned_fit(messages, (system, request, messages[-1]), budget=1100)
    result = next(m for m in fitted if m["content"].startswith("TOOL RESULT"))
    assert "shortened" not in result["content"] and result["content"].count("KEEPME") == 150
    assert any(m["content"] == "THE QUESTION" for m in fitted)
    assert not any("old chat 0 " in m["content"] for m in fitted)          # the oldest history went
    assert any("Older messages" in n for n in notes)


def test_this_turns_results_are_shortened_only_when_nothing_older_is_left():
    start, end = A.UNTRUSTED_START, A.UNTRUSTED_END
    system = {"role": "system", "content": "S"}
    request = {"role": "user", "content": "Q"}
    big = lambda tag: {"role": "user", "content": f"TOOL RESULT\n\n{start}\n" + (tag + " ") * 900 + f"\n{end}"}
    messages = [system, request, big("AAA"), big("BBB")]
    fitted, _ = A.pinned_fit(messages, (system, request, messages[-1]), budget=900)
    results = [m["content"] for m in fitted if m["content"].startswith("TOOL RESULT")]
    assert "shortened" in results[0] and "BBB" in results[-1] and "shortened" not in results[-1]


def test_prefetch_context_check_ignores_trimmed_results_from_older_turns():
    from types import SimpleNamespace
    from agent.agent import prefetch_context_intact
    state = SimpleNamespace(messages=[
        {"role": "user", "content": "TOOL RESULT\n[older tool result shortened to save space]"},
        {"role": "user", "content": "new request"},
        {"role": "user", "content": "TOOL RESULT\ncurrent complete source"},
    ])
    assert prefetch_context_intact(state, 1)


def test_prefetch_context_check_rejects_a_trimmed_current_turn_result():
    from types import SimpleNamespace
    from agent.agent import prefetch_context_intact
    state = SimpleNamespace(messages=[
        {"role": "user", "content": "TOOL RESULT\ncurrent source\ncharacters omitted because they do not fit"},
    ])
    assert not prefetch_context_intact(state, 1)
