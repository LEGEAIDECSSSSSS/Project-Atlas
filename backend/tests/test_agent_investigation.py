"""Agent protocol and investigation: parsing, duplicates, route/startup tracing, pagination notes."""
import json

from agent import agent as A
from tests.conftest import SAMPLE_PROJECT
from tests.test_agent_gate import World, run


def sent(out, index=0):
    """Everything the model was shown on request `index`, as plain text."""
    return "\n".join(m["content"] for m in out.model.calls[index])


def tool_json(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


# ---------------------------------------------------------------------------- parsing
def test_json_documents_are_answers_not_tool_calls():
    out = run("give me a json person", ['{"name": "Ada", "age": 36}'], project_id=None)
    assert out.text == '{"name": "Ada", "age": 36}' and out.streamed == out.text


def test_truncated_tool_calls_are_still_recognised_as_attempts():
    assert A.looks_like_tool_attempt('{"tool": "search_proj')
    assert not A.looks_like_tool_attempt('{"name": "Ada", "age": 36}')
    out = run("hello there", ['{"tool": "search_project", "argum', "fixed: hello"], project_id=None)
    assert out.text == "fixed: hello" and "INVALID TOOL CALL" in sent(out, 1)


def test_several_tool_calls_in_one_message_run_one_and_say_so(project):
    w = project(SAMPLE_PROJECT)
    two = tool_json("find_symbol", name="find_user") + "\n" + tool_json("find_symbol", name="health")
    out = run("what does the app do overall?", [two, "Done."], world=w)
    assert out.text == "Done."
    assert "only the first was run" in sent(out, 1)


def test_prose_around_a_tool_call_does_not_hide_it(project):
    w = project(SAMPLE_PROJECT)
    out = run("what does the app do overall?",
              ["I will look.\n```json\n" + tool_json("find_entry_points") + "\n```\nthen wait", "Done."], world=w)
    assert out.text == "Done." and any(t["name"] == "find_entry_points" for t in out.tools)


# ---------------------------------------------------------------------------- protocol
def test_unknown_arguments_are_reported_back_not_dropped_silently(project):
    w = project(SAMPLE_PROJECT)
    call = tool_json("find_symbol", name="find_user", colour="red", project_id="x")
    out = run("what does the app do overall?", [call, "Done."], world=w)
    seen = sent(out, 1)
    assert "Ignored argument(s) colour" in seen and "project_id" not in seen.split("Ignored argument(s)")[1][:40]


def test_an_identical_repeat_is_refused_and_not_counted_as_progress(project):
    w = project(SAMPLE_PROJECT)
    call = tool_json("find_symbol", name="find_user")
    out = run("what does the app do overall?", [call, call, "Done."], world=w)
    oks = [t["ok"] for t in out.tools][-2:]
    assert oks == [True, False] and "Duplicate call" in sent(out, 2)


def test_circling_searches_get_a_stop_searching_hint(project):
    w = project(SAMPLE_PROJECT)
    calls = [tool_json("find_symbol", name="find_user"), tool_json("find_symbol", name="find_use"),
             tool_json("find_symbol", name="find_us"), tool_json("find_symbol", name="find_u"), "Done."]
    out = run("what does the app do overall?", calls, world=w)
    assert any("showed nothing new" in "".join(m["content"] for m in c) for c in out.model.calls)


def test_bad_arguments_get_a_specific_message(project):
    w = project(SAMPLE_PROJECT)
    out = run("what does the app do overall?", [tool_json("read_symbol"), "Done."], world=w)
    assert "Invalid arguments for read_symbol" in sent(out, 1) and "or both" in sent(out, 1)


def test_index_tools_resolve_short_paths_and_reject_traversal(project):
    w = project(SAMPLE_PROJECT)
    out = run("what does the app do overall?",
              [tool_json("find_imports", path="auth_service.py"), tool_json("find_imports", path="../../etc/passwd"), "Done."],
              world=w)
    ok = [t["ok"] for t in out.tools][-2:]
    assert ok == [True, False] and "which is not allowed" in sent(out, 2)


# ---------------------------------------------------------------------------- investigation
def test_endpoint_questions_trace_the_route_and_read_the_chain(project):
    w = project(SAMPLE_PROJECT)
    out = run("How does the /login endpoint work?", ["It calls authenticate_user."], mode="code", world=w)
    labels = [t["label"] for t in out.tools]
    assert "Traced route /login" in labels
    reads = [l for l in labels if l.startswith("Read ")]
    assert any("authenticate_user" in l or "login" in l for l in reads)
    prompt = sent(out)
    assert "def authenticate_user" in prompt and "frontend" not in prompt.split("ATLAS INVESTIGATION")[0][:0]
    assert out.text == "It calls authenticate_user."


def test_startup_questions_find_entry_points(project):
    w = project(SAMPLE_PROJECT)
    out = run("How does the app start up?", ["It runs main()."], mode="code", world=w)
    assert any(t["name"] == "find_entry_points" for t in out.tools)
    assert "backend/main.py" in sent(out)


def test_caller_and_test_questions_use_the_graph_tools(project):
    w = project(SAMPLE_PROJECT)
    try:
        out = run("who calls authenticate_user?", ["login and a test."], mode="code", world=w,
                  close=False)
        assert any(t["name"] == "find_callers" for t in out.tools)
        out = run("which tests cover authenticate_user?", ["test_login_ok."], mode="code", world=w,
                  close=False)
        assert any(t["name"] == "find_tests_for_symbol" for t in out.tools)
    finally:
        w.close()


def test_vague_questions_start_from_the_entry_points(project):
    w = project(SAMPLE_PROJECT)
    out = run("check my code", ["Looks fine."], mode="code", world=w)
    assert any(t["name"] == "find_entry_points" for t in out.tools)
    assert any(t["label"].startswith("Read backend/main.py") for t in out.tools)


def test_small_talk_and_conceptual_questions_trigger_no_investigation(project):
    w = project(SAMPLE_PROJECT)
    for text in ["hello there", "thanks!", "What are the pros and cons of microservices?"]:
        out = run(text, ["ok"], mode="code", world=w)
        assert out.tools == [] and out.model_calls == 1


def test_the_investigation_stays_inside_its_budget(project, monkeypatch):
    import config
    big = {f"m{i}.py": "def target_fn():\n" + "    x = 1\n" * 300 for i in range(8)}
    w = project(big)
    out = run("Explain target_fn", ["ok"], mode="code", world=w)
    used = sum(len(m["content"]) for m in out.model.calls[0] if m["role"] == "user")
    assert used < config.PROMPT_BUDGET_TOKENS * config.CHARS_PER_TOKEN


def test_one_search_shows_several_ranked_files_with_complete_definitions(project):
    files = {f"mod{i}.py": f"import os\n\ndef compute_total():\n    return {i}\n" + "# pad\n" * 600 for i in range(3)}
    w = project(files)
    out = run("find where compute_total is defined", ["It is in the modules."], mode="code", world=w)
    prompt = sent(out)
    assert prompt.count("BEGIN FILE: mod") >= 2
    assert "def compute_total" in prompt and "complete" in prompt


def test_search_results_say_when_more_pages_exist(project):
    files = {f"f{i:02d}.py": "needle = 1\n" for i in range(30)}
    w = project(files)
    out = run("find where needle is used", ["Many places."], mode="code", world=w)
    prompt = sent(out)
    assert "Showing 8 of 30 matching files" in prompt and "offset=8" in prompt


def test_the_model_can_page_forward(project):
    files = {f"f{i:02d}.py": "needle = 1\n" for i in range(30)}
    w = project(files)
    page2 = tool_json("search_project", query="needle", offset=8, limit=8)
    out = run("find where needle is used", [page2, "Done."], mode="code", world=w)
    assert out.text == "Done." and "starting at 8" in sent(out, 1)


def test_negative_results_are_honest_after_two_empty_searches(project):
    w = project(SAMPLE_PROJECT)
    out = run("Where is the refund_payment function?", ["I could not find refund_payment."], mode="code", world=w)
    assert out.model_calls == 1 and "Nothing matching was found" in sent(out)


def test_follow_up_messages_still_work_with_history(project):
    w = project(SAMPLE_PROJECT)
    history = [{"role": "user", "content": "How does login work?"}, {"role": "assistant", "content": "It checks credentials."}]
    out = run("and what about find_user?", ["It reads USERS."], mode="code", world=w, history=history)
    assert out.text == "It reads USERS." and "How does login work?" in sent(out)


def test_investigation_survives_a_broken_index(project, monkeypatch):
    w = project(SAMPLE_PROJECT)
    from services import code_index
    monkeypatch.setattr(code_index, "get_index", lambda pid: (_ for _ in ()).throw(RuntimeError("boom")))
    out = run("How does the /login endpoint work?", ["I could not trace it."], mode="code", world=w)
    assert out.model_calls >= 1 and out.done["type"] == "done"


TASKBOARD_LOGIN_FLOW = {
    "frontend/src/pages/Login.jsx": """import { useAuth } from '../auth'

export default function Login() {
  const { login } = useAuth()

  async function submit(event) {
    event.preventDefault()
    await login('person@example.com', 'password')
  }
  return <form onSubmit={submit} />
}
""",
    "frontend/src/auth.jsx": """import { useCallback, useContext } from 'react'
import { api } from './api'
const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const login = useCallback(
    async (email, password) => {
      const me = await api.post('/auth/login', { email, password })
      return me
    },
    [],
  )
  const value = useMemo(() => ({ login }), [login])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export const useAuth = () => useContext(AuthContext)
""",
    "frontend/src/api.js": """const BASE = '/api'

async function request(method, path, body) {
  return fetch(BASE + path, {
    method,
    credentials: 'include',
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

export const api = {
  post: (path, body) => request('POST', path, body),
}
""",
    "app/controllers/auth_controller.py": """from fastapi import APIRouter, Request, Response
from app.dependencies import AuthServiceDep

router = APIRouter(prefix='/api/auth')


def set_auth_cookies(response, access, refresh):
    response.set_cookie('access', access, httponly=True, path='/api')
    response.set_cookie('refresh', refresh, httponly=True, path='/api/auth')


def user_out(user):
    return {'id': user.id, 'email': user.email}


@router.post('/login')
def login(body, request: Request, response: Response, service: AuthServiceDep):
    user, access, refresh = service.login(body, request.headers.get('user-agent'))
    set_auth_cookies(response, access, refresh)
    return user_out(user)
""",
    "app/services/auth_service.py": """from app.repositories.user_repository import UserRepository
from app.core.security import verify_password, create_access_token, generate_token, hash_token

class AuthService:
    def login(self, request, user_agent):
        user = self.users.find_by_email(request.email)
        if not verify_password(request.password, user.password_hash if user else None) or user is None:
            raise ValueError('invalid credentials')
        refresh = self._start_session(user, user_agent)
        return user, create_access_token(user.id), refresh

    def _start_session(self, user, user_agent):
        token = generate_token()
        self.sessions.create(user.id, hash_token(token), user_agent)
        return token
""",
    "app/repositories/user_repository.py": """class UserRepository:
    def find_by_email(self, email):
        return self.session.exec(select(User).where(User.email == email.lower())).first()

class SessionRepository:
    def create(self, user_id, token_hash, user_agent):
        return self.add(RefreshSession(user_id=user_id, token_hash=token_hash, user_agent=user_agent))
""",
    "app/core/security.py": """def verify_password(plain, hashed):
    return password_context.verify(plain, hashed)

def generate_token():
    return secrets.token_urlsafe(32)

def hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()

def create_access_token(user_id):
    return jwt.encode({'sub': str(user_id)}, SECRET, algorithm=ALGORITHM)
""",
}

FLOW_TEST_ANSWER = """**Function:** `submit`
**File Path:** `frontend/src/pages/Login.jsx`
**Line Numbers:** 6-9

**Function:** `login`
**File Path:** `frontend/src/auth.jsx`
**Line Numbers:** 6-10

**Function:** `post`
**File Path:** `frontend/src/api.js`
**Line Numbers:** 12-12

**Function:** `request`
**File Path:** `frontend/src/api.js`
**Line Numbers:** 3-9

**Function:** `login`
**File Path:** `app/controllers/auth_controller.py`
**Line Numbers:** 16-20

**Function:** `login`
**File Path:** `app/services/auth_service.py`
**Line Numbers:** 5-10

**Function:** `find_by_email`
**File Path:** `app/repositories/user_repository.py`
**Line Numbers:** 2-3

**Function:** `verify_password`
**File Path:** `app/core/security.py`
**Line Numbers:** 1-2

**Function:** `_start_session`
**File Path:** `app/services/auth_service.py`
**Line Numbers:** 12-15

**Function:** `create_access_token`
**File Path:** `app/core/security.py`
**Line Numbers:** 10-11

**Function:** `set_auth_cookies`
**File Path:** `app/controllers/auth_controller.py`
**Line Numbers:** 7-9

**Function:** `user_out`
**File Path:** `app/controllers/auth_controller.py`
**Line Numbers:** 12-13"""

FLOW_PROMPT = (
    "Trace the complete login process in this project, starting from the frontend login page "
    "and following the request through the backend until the user is authenticated and the "
    "response is returned. Identify the relevant functions and files involved. Explain the "
    "sequence of calls and the responsibility of each function. Include the frontend function "
    "that initiates the login request, the API client function that sends the request, the "
    "backend route handler, the authentication service method, how the user is located and "
    "password verified, and how tokens and session cookies are created and returned. For every "
    "function, provide its file path and line numbers. Use actual project source as evidence. "
    "Do not invent missing implementation details."
)


def test_complete_login_flow_prefetches_frontend_and_backend_sources_end_to_end(project):
    w = project(TASKBOARD_LOGIN_FLOW)
    out = run(FLOW_PROMPT, [FLOW_TEST_ANSWER], mode="code", world=w)
    labels = [tool["label"] for tool in out.tools]
    assert any(label.startswith("Traced route login") for label in labels)
    reads = {
        (record.get("path"), record.get("lines"))
        for record in out.done["evidence"]
        if record.get("tool") in ("read_project_file", "read_symbol")
    }
    expected_paths = {
        "frontend/src/pages/Login.jsx", "frontend/src/auth.jsx", "frontend/src/api.js",
        "app/controllers/auth_controller.py", "app/services/auth_service.py",
        "app/repositories/user_repository.py", "app/core/security.py",
    }
    assert {path for path, _lines in reads} == expected_paths
    assert all(record.get("complete") for record in out.done["evidence"]
               if record.get("tool") in ("read_project_file", "read_symbol"))
    prompt = sent(out)
    for required in (
        "async function submit", "const { login } = useAuth()", "api.post('/auth/login'",
        "async function request", "APIRouter(prefix='/api/auth')", "def login(body",
        "def login(self, request", "def find_by_email", "def verify_password",
        "def _start_session", "def create_access_token", "def set_auth_cookies", "def user_out",
    ):
        assert required in prompt, required
    assert "POST /api/auth/login" in prompt
    assert "unverified draft was withheld" not in out.text



def test_flow_trace_reads_project_specific_helpers_and_typescript_api_clients(project):
    files = {path: source.replace("useAuth", "useSession")
             for path, source in TASKBOARD_LOGIN_FLOW.items()}
    files["frontend/src/api.ts"] = files.pop("frontend/src/api.js")
    service = files["app/services/auth_service.py"]
    service = service.replace(
        "        user = self.users.find_by_email(request.email)\n",
        "        request = self.prepare_login(request)\n"
        "        user = self.users.find_by_email(request.email)\n",
    )
    service += (
        "\n    def prepare_login(self, request):\n"
        "        request.email = request.email.strip().lower()\n"
        "        return request\n"
    )
    files["app/services/auth_service.py"] = service

    w = project(files)
    out = run(FLOW_PROMPT, [FLOW_TEST_ANSWER], mode="code", world=w)
    prompt = sent(out)
    assert "def prepare_login(self, request)" in prompt
    assert "request.email = request.email.strip().lower()" in prompt
    assert "const BASE = '/api'" in prompt
    assert "useSession" in prompt
    assert any(
        record.get("path") == "frontend/src/api.ts" and record.get("complete")
        for record in out.done["evidence"]
        if record.get("tool") in ("read_project_file", "read_symbol")
    )


def test_flow_trace_preserves_core_backend_evidence_in_the_final_model_prompt(project):
    """A flow investigation must not spend its prompt budget on UI preambles first.

    This exercises the real request -> route graph -> prioritized source reads -> final
    model prompt path. It guards against the previous behaviour where early tool results
    were shortened to 300 characters before the backend service was ever shown to the model.
    """
    w = project(TASKBOARD_LOGIN_FLOW)
    out = run(FLOW_PROMPT, [FLOW_TEST_ANSWER], mode="code", world=w)

    records = [record for record in out.done["evidence"]
               if record.get("tool") in ("read_project_file", "read_symbol")]
    paths = {record.get("path") for record in records}
    required = {
        "frontend/src/pages/Login.jsx", "frontend/src/auth.jsx", "frontend/src/api.js",
        "app/controllers/auth_controller.py", "app/services/auth_service.py",
        "app/repositories/user_repository.py", "app/core/security.py",
    }
    assert required <= paths
    assert all(record.get("complete") for record in records)

    final_prompt = "\n".join(message["content"] for message in out.model.calls[-1])
    assert "[older tool result shortened" not in final_prompt
    assert "ATLAS VERIFIED FLOW EVIDENCE" in final_prompt
    assert "VERIFIED DEFINITIONS" in final_prompt
    assert "VERIFIED CALL SITES" in final_prompt
    assert "SOURCE app/services/auth_service.py | lines 5-15 | COMPLETE" in final_prompt
    assert "TOOL RESULT" not in final_prompt
    for source_marker in (
        "async function submit", "async (email, password)", "async function request",
        "def login(body", "def login(self", "find_by_email", "verify_password",
        "_start_session", "create_access_token", "set_auth_cookies", "user_out",
    ):
        assert source_marker in final_prompt, source_marker


def test_flow_investigation_keeps_a_reserve_for_answer_generation():
    from agent import agent as A

    assert A.FLOW_INVESTIGATION_SHARE <= 0.75
    assert A.MAX_FLOW_READS <= 18


def test_flow_trace_strict_verifier_withholds_invented_auth_helpers(project):
    w = project(TASKBOARD_LOGIN_FLOW)
    hallucinated = (
        "The route calls **_verify_password**, **_create_access_token**, "
        "**_create_refresh_token**, and **_set_auth_cookies**."
    )
    out = run(FLOW_PROMPT, [hallucinated], mode="code", world=w)
    assert "unverified draft was withheld" in out.text
    assert "_verify_password" in out.text
    assert "_create_refresh_token" in out.text


def test_login_route_graph_joins_router_prefix_frontend_client_and_real_service(project):
    from services import code_index

    w = project(TASKBOARD_LOGIN_FLOW)
    trace = code_index.trace_route(w.project_id, "/api/auth/login", depth=4)
    assert len(trace["routes"]) == 1
    route = trace["routes"][0]
    assert route["effective_route"] == "/api/auth/login"
    assert route["declared_at"].startswith("app/controllers/auth_controller.py:")
    assert route["frontend_callers"] == [{
        "path": "frontend/src/auth.jsx", "line": 8, "method": "POST",
        "request_path": "/auth/login", "symbol": "login", "start_line": 6, "end_line": 10,
    }]
    backend_symbols = {node["symbol"] for node in route["chain"]}
    assert "AuthService.login" in backend_symbols
    assert "AuthService._start_session" in backend_symbols
    assert "UserRepository.find_by_email" in backend_symbols
    assert "verify_password" in backend_symbols
    assert "create_access_token" in backend_symbols
    assert "login" not in {
        node["symbol"] for node in route["chain"][1:]
        if node.get("path", "").startswith("frontend/")
    }
    api_chain_symbols = {
        node["symbol"] for chain in route["frontend_chains"] for node in chain["chain"]
    }
    assert "api.post" in api_chain_symbols and "request" in api_chain_symbols
    assert any(item["symbol"] == "submit" for item in route["frontend_ui_callers"])
    assert not any(path.startswith("frontend/") and parsed.path == "/auth/login"
                   for path, parsed in code_index.get_index(w.project_id).routes)


def test_flow_trace_discovers_frontend_under_an_archive_root_and_reads_requested_backend_layers(project):
    """An archive root and nonstandard source folders must not hide client/server files."""
    wrapped = {}
    for path, source in TASKBOARD_LOGIN_FLOW.items():
        path = path.replace("frontend/src", "client_ui/code").replace("app/", "server_src/")
        source = source.replace("from app.", "from server_src.")
        wrapped[f"arbitrary-upload-root/{path}"] = source
    w = project(wrapped)
    wrapped_answer = FLOW_TEST_ANSWER.replace("frontend/src", "client_ui/code")
    wrapped_answer = wrapped_answer.replace("app/", "server_src/")
    wrapped_answer = wrapped_answer.replace("client_ui/code", "arbitrary-upload-root/client_ui/code")
    wrapped_answer = wrapped_answer.replace("server_src/", "arbitrary-upload-root/server_src/")
    out = run(FLOW_PROMPT, [wrapped_answer], mode="code", world=w)

    assert any(tool["name"] == "trace_route" for tool in out.tools)
    read_records = [record for record in out.done["evidence"]
                    if record.get("tool") in ("read_project_file", "read_symbol")]
    read_paths = {record.get("path") for record in read_records}
    assert "arbitrary-upload-root/client_ui/code/pages/Login.jsx" in read_paths
    assert "arbitrary-upload-root/client_ui/code/auth.jsx" in read_paths
    assert "arbitrary-upload-root/client_ui/code/api.js" in read_paths
    assert "arbitrary-upload-root/server_src/controllers/auth_controller.py" in read_paths
    assert "arbitrary-upload-root/server_src/services/auth_service.py" in read_paths
    assert "arbitrary-upload-root/server_src/repositories/user_repository.py" in read_paths
    assert "arbitrary-upload-root/server_src/core/security.py" in read_paths
    assert all(record.get("complete") for record in read_records)
    assert "unverified draft was withheld" not in out.text


def test_route_graph_uses_receiver_types_to_disambiguate_same_named_methods(project):
    from services import code_index

    files = dict(TASKBOARD_LOGIN_FLOW)
    files["app/services/auth_service.py"] = files["app/services/auth_service.py"].replace(
        "from app.repositories.user_repository import UserRepository",
        "from app.repositories.user_repository import UserRepository, SessionRepository, AuthTokenRepository",
    ).replace(
        "class AuthService:\n",
        "class AuthService:\n"
        "    def __init__(self, users, sessions):\n"
        "        self.users = UserRepository(users)\n"
        "        self.sessions = SessionRepository(sessions)\n\n",
    )
    files["app/repositories/user_repository.py"] += (
        "\nclass AuthTokenRepository:\n"
        "    def create(self, user_id, token_hash):\n"
        "        return self.add((user_id, token_hash))\n"
    )
    w = project(files)
    route = code_index.trace_route(w.project_id, "/api/auth/login", depth=4)["routes"][0]
    symbols = [node["symbol"] for node in route["chain"]]
    assert "SessionRepository.create" in symbols
    assert "AuthTokenRepository.create" not in symbols


def test_backend_javascript_routes_are_not_mistaken_for_frontend_api_callers(project):
    from services import code_index

    files = dict(TASKBOARD_LOGIN_FLOW)
    files["server/routes.js"] = '''
const app = makeServer()
app.post('/api/auth/login', async (request, response) => {
  return fetch('/api/auth/login', { method: 'POST' })
})
'''
    w = project(files)
    routes = code_index.trace_route(w.project_id, "/api/auth/login", depth=2)["routes"]
    python_route = next(route for route in routes if route["framework"].startswith("fastapi"))
    assert all(item["path"] != "server/routes.js" for item in python_route["frontend_callers"])



def test_flow_prompt_excludes_imported_library_apis_from_project_function_references():
    from agent.prompts import FLOW_TRACE_SYSTEM_PROMPT

    assert "List project-defined functions, methods and callbacks" in FLOW_TRACE_SYSTEM_PROMPT
    assert "Do not present imported framework/library APIs" in FLOW_TRACE_SYSTEM_PROMPT
    assert "Do not estimate line numbers" in FLOW_TRACE_SYSTEM_PROMPT
