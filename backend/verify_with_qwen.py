"""Verify the Mini Trello login trace against the REAL Ollama model.

    python verify_with_qwen.py path/to/taskboard.zip

Runs the real Atlas pipeline (classification, search, index, retrieval, evidence packet, call
analysis, real model generation, verifier) in an isolated temporary data directory, so it never
touches your conversations or workspace.

Every check is labelled by what actually ran:
  [REAL MODEL]    needs Ollama + the model; reported NOT PERFORMED if Ollama is unreachable.
  [NO MODEL]      deterministic parts (single-file retrieval, fabricated-reference rejection). A scripted
                  stand-in replaces the model here, and it is NOT Qwen.

The expected definitions below are specific to the Mini Trello archive and were checked by hand
against its source. They are test data for this script only; Atlas itself contains no project names.
"""
import asyncio
import io
import os
import re
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="atlas-verify-")
os.environ["LC_DATA_DIR"] = _tmp
os.environ.setdefault("LC_WARM_INDEX", "0")

import httpx  # noqa: E402

import config  # noqa: E402
from agent import agent as A  # noqa: E402
from agent import verify as V  # noqa: E402
from services import project_service  # noqa: E402

EXPECTED = {                                     # (function, project-relative path) -> (start, end)
    ("submit", "frontend/src/pages/Login.jsx"): (27, 44), ("login", "frontend/src/auth.jsx"): (38, 44),
    ("api.post", "frontend/src/api.js"): (72, 72), ("request", "frontend/src/api.js"): (35, 68),
    ("login", "app/controllers/auth_controller.py"): (65, 72),
    ("_client_ip", "app/controllers/auth_controller.py"): (19, 20),
    ("set_auth_cookies", "app/controllers/auth_controller.py"): (23, 28),
    ("user_out", "app/controllers/auth_controller.py"): (36, 43),
    ("AuthService.login", "app/services/auth_service.py"): (84, 94),
    ("AuthService._start_session", "app/services/auth_service.py"): (123, 131),
    ("UserRepository.find_by_email", "app/repositories/user_repository.py"): (16, 17),
    ("SessionRepository.create", "app/repositories/user_repository.py"): (44, 47),
    ("verify_password", "app/core/security.py"): (21, 25), ("generate_token", "app/core/security.py"): (28, 29),
    ("hash_token", "app/core/security.py"): (32, 34), ("create_access_token", "app/core/security.py"): (37, 45),
}
HEADING = "### Atlas-verified connections"


def V_conn(entry):
    """The Connection part of a parsed entry (lower-case)."""
    block = (entry or {}).get("block", "")
    i = block.lower().find("connection")
    return block[i:] if i >= 0 else ""


FLOW_PROMPT = (
    "Trace the complete login process in this project, starting from the frontend login page and following "
    "the request through the backend until the user is authenticated and the response is returned.\n\n"
    "Identify the relevant functions and files involved. Explain the sequence of calls and the responsibility "
    "of each function.\n\nInclude:\n1. The frontend function that initiates the login request.\n"
    "2. The API client function that sends the request.\n"
    "3. The backend route handler, including its complete signature.\n"
    "4. The authentication service method called by the route.\n"
    "5. How the user is located and how the password is verified.\n"
    "6. How authentication tokens and session cookies are created and returned.\n\n"
    "For every function, provide its file path and line numbers.\n\n"
    "Use the actual project source as evidence. Do not invent missing implementation details. "
    "Do not return entire files. If a detail cannot be verified from the available source, explicitly say so."
)
SINGLE_FILE_PROMPT = (
    "Inspect the login function in app/controllers/auth_controller.py. Return the complete function signature, "
    "complete function body, and starting and ending line numbers. Do not read or return the entire file.")

results = []


def record(status, label, detail=""):
    results.append(status)
    print(f"  [{status}] {label}" + (f"\n         {detail}" if detail else ""))


class Upload:
    def __init__(self, name, data):
        self.filename, self.file = name, io.BytesIO(data)


async def run(project_id, message, recorder=None):
    events = []
    async for event in A.run_agent_stream(message=message, history=[], project_id=project_id, mode="code",
                                          model=config.CODE_MODEL, project_context="", evidence=[],
                                          tests_approved=False):
        events.append(event)
    return events


def final_text(events):
    done = next((e for e in events if e["type"] == "done"), None)
    return (done or {}).get("response", "")


def withheld(text):
    return text.startswith("Atlas could not provide a verified source-evidence answer")


def ollama_up():
    try:
        tags = httpx.get(config.OLLAMA_URL + "/api/tags", timeout=3).json()
        names = {m.get("name") for m in tags.get("models", [])}
        return (config.CODE_MODEL in names) or any(n and n.split(":")[0] == config.CODE_MODEL.split(":")[0] for n in names)
    except Exception:
        return False


def entry_for(entries, name, path):
    for e in entries:
        if (e.get("function") or "").split(".")[-1] == name.split(".")[-1] and (e.get("path") or "").endswith(path):
            return e
    return None


def main(zip_path, prompt_file=None):
    global FLOW_PROMPT
    if prompt_file:                     # e.g. tests/fixtures/prompt_latest_user.txt, the wording that once skipped the trace
        with open(prompt_file, encoding="utf-8") as handle:
            FLOW_PROMPT = handle.read().strip()
    with open(zip_path, "rb") as handle:
        stats = project_service.extract_project_with_stats(Upload(os.path.basename(zip_path), handle.read()))
    pid = stats["project_id"]
    root = ""
    print(f"Atlas verification | model {config.CODE_MODEL} | NUM_CTX {config.NUM_CTX} | project {pid[:8]}"
          + (f" | prompt {os.path.basename(prompt_file)}" if prompt_file else ""))

    # ------------------------------------------------------------------ [NO MODEL] single-file retrieval
    print("\n[NO MODEL] single-file symbol retrieval")
    text = final_text(asyncio.run(run(pid, SINGLE_FILE_PROMPT)))
    ok = "65" in text and "72" in text and "def login(" in text and "def logout" not in text
    record("PASS" if ok else "FAIL", "login returned with its signature and body, lines 65-72, no unrelated code")

    # ------------------------------------------------------------------ [NO MODEL] verifier rejects fabrications
    print("\n[NO MODEL] fabricated references are rejected (scripted stand-in for the model, NOT Qwen)")
    prefix = next((p for p in A.listing_paths(project_service.list_project_files(pid))
                   if p.endswith("app/controllers/auth_controller.py")), "app/controllers/auth_controller.py")
    root = prefix[: -len("app/controllers/auth_controller.py")]

    def entry(fn, path, a, b, conn):
        return (f"**Function:** {fn}\n**File Path:** {root}{path}\n**Line Numbers:** {a}-{b}\n**Purpose:** x\n"
                f"**Connection:** {conn}\n\n")

    fabrications = {
        "wrong line range": entry("login", "app/controllers/auth_controller.py", 66, 73, "calls `service.login(...)`"),
        "invented function": entry("hash_password", "app/core/security.py", 10, 20, "calls `bcrypt.hash(...)`"),
        "fabricated call": entry("login", "app/controllers/auth_controller.py", 65, 72, "calls `verify_password(...)` at line 70"),
        "call on the wrong line": entry("login", "app/controllers/auth_controller.py", 65, 72, "calls `service.login(...)` at line 66"),
    }
    real = A.stream_chat
    for name, answer in fabrications.items():
        async def stub(messages, model, temperature, num_predict=None, _a=answer, **kw):
            yield {"text": _a}
            yield {"done": True, "stats": {"tokens": 10, "tokens_per_second": 1.0, "prompt_tokens": None}}
        A.stream_chat = stub
        try:
            rejected = withheld(final_text(asyncio.run(run(pid, FLOW_PROMPT))))
        finally:
            A.stream_chat = real
        record("PASS" if rejected else "FAIL", f"rejected: {name}")

    # ------------------------------------------------------------------ [REAL MODEL]
    print("\n[REAL MODEL] login trace with the actual model")
    if not ollama_up():
        record("NOT PERFORMED", f"Ollama not reachable at {config.OLLAMA_URL} with model {config.CODE_MODEL}",
               "Start Ollama, run `ollama pull " + config.CODE_MODEL + "`, then rerun this script.")
        return summary()

    sent = []
    original = A.stream_chat

    async def recording(messages, model, temperature, num_predict=None, **kw):
        sent.append([dict(m) for m in messages])
        async for chunk in original(messages, model, temperature, num_predict=num_predict, **kw):
            yield chunk
    A.stream_chat = recording
    try:
        events = asyncio.run(run(pid, FLOW_PROMPT))
    finally:
        A.stream_chat = original
    answer = final_text(events)
    done = next((e for e in events if e["type"] == "done"), {})
    packet = sent[-1][-1]["content"] if sent else ""
    print(f"  model calls: {len(sent)} | packet: {len(packet)} chars | stats: {done.get('stats')}")
    record("PASS" if "VERIFIED DEFINITIONS" in packet else "FAIL", "the model was sent a verified, numbered evidence packet")
    record("PASS" if not withheld(answer) else "FAIL", "the answer passed the evidence verifier (not withheld)",
           answer[:600].replace("\n", " ") if withheld(answer) else "")
    if withheld(answer):
        print("\n  The verifier rejected the model's draft, so the content checks below are not meaningful.")
        print("  This is the verifier doing its job: the reasons above say which claim the source does not support.")
        return summary()

    entries = V.parse_function_entries(answer)
    missing, wrong = [], []
    for (name, path), (start, end) in EXPECTED.items():
        hit = entry_for(entries, name, path)
        if hit is None:
            missing.append(f"{name} ({path})")
        elif (hit["start"], hit["end"]) != (start, end):
            wrong.append(f"{name}: said {hit['start']}-{hit['end']}, expected {start}-{end}")
    record("PASS" if not wrong else "FAIL", "every reported definition has the correct inclusive line range", "; ".join(wrong))
    record("PASS" if not missing else "FAIL", f"all {len(EXPECTED)} expected definitions are reported", "missing: " + ", ".join(missing))
    extras = [e["function"] for e in entries if not any(entry_for([e], n, p) for n, p in EXPECTED)]
    record("PASS" if not extras else "FAIL", "no unrelated definitions were added", ", ".join(map(str, extras)))
    restored = len(re.findall(r"^\*\*Function:\*\* ", answer.split("### Not covered", 1)[1], re.M)) if "### Not covered" in answer else 0
    print(f"  info: the model itself wrote {len(entries) - restored} entries; Atlas restored {restored} it skipped")

    def combined(name, path):
        """The model's entry plus the Atlas-verified bullet for the same function (lower-cased)."""
        e = entry_for(entries, name, path)
        text = (e or {}).get("block", "")
        for heading, stop in (("### Atlas corrections to the answer above", "### Atlas-verified connections"),
                              (HEADING, "### Not covered")):
            if heading in answer:
                section = answer.split(heading, 1)[1].split(stop, 1)[0]
                for m in re.finditer(r"- \*\*`" + re.escape(name) + r"`\*\*.*?(?=\n- \*\*`|\Z)", section, re.S):
                    text += "\n" + m.group(0)
        return text.lower()

    unsure = re.compile(r"cannot be (?:established|verified)|could not be verified", re.I)
    still = [n for (n, p) in EXPECTED if unsure.search(V_conn(entry_for(entries, n, p))) and
             not (HEADING in answer and re.search(r"- \*\*`" + re.escape(n) + r"`\*\*", answer))]
    record("PASS" if not still else "FAIL", "no connection is left as 'cannot be established' without an Atlas-verified fact",
           ", ".join(still))
    vp = combined("verify_password", "app/core/security.py")
    record("PASS" if "argon" in vp and "called by" in vp or "authservice.login" in vp and "argon" in vp else "FAIL",
           "verify_password: called by AuthService.login, and _hasher.verify is an external argon2 call")
    rq = combined("request", "frontend/src/api.js")
    record("PASS" if "fetch" in rq and "refreshsession" in rq and "recurs" in rq and "condition" in rq else "FAIL",
           "request: fetch (platform API), conditional refreshSession(), conditional recursive request()")
    sess = combined("AuthService._start_session", "app/services/auth_service.py")
    record("PASS" if "self.sessions.create" in sess and "sessionrepository" in sess else "FAIL",
           "_start_session: self.sessions.create(...) shown, receiver resolved to SessionRepository")
    ctl = combined("login", "app/controllers/auth_controller.py")
    record("PASS" if all(w in ctl for w in ("access", "refresh", "set_auth_cookies", "user_out", "body", "response")) else "FAIL",
           "controller login: access/refresh to set_auth_cookies, user to user_out, body/request/response/service roles")
    cai = combined("create_access_token", "app/core/security.py")
    record("PASS" if "generate_token" not in cai else "FAIL",
           "create_access_token: does not claim a call to generate_token (it does not make one)")
    # ---- does Atlas challenge wrong claims? (the model's text itself is never rewritten)
    CORR = "### Atlas corrections to the answer above"
    corrections = answer.split(CORR, 1)[1].split("### Atlas-verified connections", 1)[0] if CORR in answer else ""
    count = len(re.findall(r"^- \*\*`", corrections, re.M))
    print(f"  info: Atlas flagged {count} claim(s) in the model's answer")
    model_text = answer.split("\n---\n", 1)[0]
    known_errors = [           # (text a model may write, what a correct challenge must say) - Mini Trello specific
        ("(line 89)", "is called at line 69"),
        ("create(user, user_agent)", "user.id, hash_token(token)"),
        ("login(email, password)", "login(normaliseEmail(email), password)"),
        ("(line 44)", "is at line 125"),
    ]
    for wrong, must_say in known_errors:
        if wrong in model_text:
            record("PASS" if must_say in corrections else "FAIL", f"the model wrote `{wrong}`: Atlas's correction says `{must_say}`")
    ctl_block = (entry_for(entries, "login", "app/controllers/auth_controller.py") or {}).get("block", "")
    claims_tokens = re.search(r"returns[^.]*\baccess\b", ctl_block) is not None
    if claims_tokens:
        record("PASS" if "`user_out(user)` (line 72)" in corrections else "FAIL",
               "the model said the route returns the tokens: Atlas states it returns user_out(user) (line 72)")
    wrongly = [l for l in corrections.splitlines() if l.startswith("- **`") and
               any(f"- **`{n}`**" in l and "**contradicted:**" in l for n in ("_client_ip", "create_access_token", "generate_token",
                                                                              "verify_password", "UserRepository.find_by_email"))]
    record("PASS" if not wrongly else "FAIL", "claims that are correct were not flagged as contradicted", "; ".join(wrongly)[:300])
    print("\n----- ANSWER FROM THE MODEL -----\n" + answer + "\n---------------------------------")
    return summary()


def summary():
    fails, skipped = results.count("FAIL"), results.count("NOT PERFORMED")
    print(f"\n{results.count('PASS')} passed, {fails} failed, {skipped} not performed")
    return 1 if fails else (2 if skipped else 0)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python verify_with_qwen.py path/to/taskboard.zip [prompt.txt]\n"
                 "  try: python verify_with_qwen.py taskboard.zip tests/fixtures/prompt_latest_user.txt")
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None))
