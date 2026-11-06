"""Words, identifiers and code vocabulary shared by project search and request planning.

Everything here is deterministic and dependency-free. The synonym table maps everyday words
to the names programmers actually use ("login" -> authenticate, sign_in, session ...), so a
question phrased in plain English can still find the code.
"""
import functools
import re

_CAMEL_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")

STOPWORDS = frozenset("""
a an the and or but if then else of to in on at by for with from into onto over under about
as is are was were be been being am do does did done doing have has had having can could
should would will shall may might must i me my mine we our ours you your yours he she it its
they them their this that these those there here what which who whom whose when where why
how any all some each every both either neither not no nor so than too very just also only
please can't cannot won't isn't aren't doesn't don't didn't let lets tell show give need
want like get got make made use used using able via per etc
""".split())

# Words that describe the *question*, not the code. They never make good search terms.
QUESTION_NOISE = frozenset("""
explain describe walk walkthrough understand overview summary summarize summarise review
check look inspect analyse analyze trace find locate search handled handles handling handle
happens happen happening work works working workings flow flows responsible code codebase
project app application program software file files folder folders thing things stuff part
parts piece way ways connect connects connected connecting called call calls calling
function functions method methods class classes module modules implemented implementation
implement implements defined define defines contain contains containing exist exists
anything something everything correct wrong broken issue issues problem problems bug bugs
fail fails failing failed fix fixes error errors why does doing
""".split())

# plain word -> names programmers use for the same idea
SYNONYMS = {
    "login": ["signin", "sign_in", "log_in", "logon", "authenticate", "authentication",
              "auth", "session", "credential", "credentials"],
    "signin": ["login", "sign_in", "authenticate", "auth", "session"],
    "logout": ["signout", "sign_out", "log_out", "logoff", "session", "revoke"],
    "signout": ["logout", "sign_out", "session"],
    "signup": ["register", "registration", "sign_up", "create_user", "enroll"],
    "register": ["signup", "registration", "sign_up", "create_user", "enroll"],
    "auth": ["authenticate", "authentication", "authorize", "authorization", "login",
             "token", "session", "jwt", "credential"],
    "authentication": ["authenticate", "auth", "login", "token", "session", "jwt"],
    "authorization": ["authorize", "permission", "role", "policy", "acl", "rbac"],
    "password": ["passwd", "pwd", "credential", "hash", "bcrypt", "secret"],
    "reset": ["forgot", "recover", "recovery", "restore"],
    "email": ["mail", "smtp", "mailer", "notification", "notify", "dispatch", "send"],
    "mail": ["email", "smtp", "mailer", "notification"],
    "notification": ["notify", "alert", "dispatch", "push", "email", "message"],
    "send": ["dispatch", "emit", "deliver", "publish", "post", "notify"],
    "start": ["startup", "boot", "bootstrap", "main", "init", "initialize", "launch", "run"],
    "startup": ["start", "boot", "bootstrap", "main", "init", "initialize", "launch",
                "lifespan"],
    "starts": ["startup", "boot", "main", "init", "launch", "run"],
    "boot": ["startup", "bootstrap", "main", "init"],
    "entry": ["main", "entrypoint", "startup", "bootstrap", "index"],
    "route": ["endpoint", "handler", "controller", "view", "router", "path", "url"],
    "router": ["route", "endpoint", "handler", "controller"],
    "endpoint": ["route", "handler", "controller", "view", "router", "api", "url"],
    "controller": ["handler", "route", "endpoint", "view"],
    "handler": ["controller", "route", "endpoint", "callback", "listener"],
    "api": ["endpoint", "route", "client", "fetch", "request", "rest"],
    "request": ["req", "http", "fetch", "call", "payload"],
    "response": ["res", "reply", "result", "payload"],
    "database": ["db", "sql", "query", "repository", "model", "orm", "connection",
                 "session", "schema", "migration"],
    "db": ["database", "sql", "repository", "orm", "connection"],
    "sql": ["query", "database", "db", "select", "insert"],
    "model": ["entity", "schema", "table", "dto"],
    "user": ["account", "profile", "member", "customer", "principal"],
    "account": ["user", "profile", "member"],
    "config": ["configuration", "settings", "setting", "env", "options", "properties"],
    "configuration": ["config", "settings", "env", "options"],
    "settings": ["config", "configuration", "env", "options", "preferences"],
    "error": ["exception", "failure", "fail", "raise", "throw", "catch"],
    "exception": ["error", "raise", "throw", "catch", "failure"],
    "test": ["spec", "unittest", "pytest", "junit", "assert", "mock", "fixture"],
    "tests": ["spec", "unittest", "pytest", "junit", "assert", "mock", "fixture"],
    "delete": ["remove", "destroy", "drop", "erase", "purge"],
    "remove": ["delete", "destroy", "drop", "discard"],
    "create": ["add", "insert", "new", "make", "build", "generate", "save"],
    "add": ["create", "insert", "append", "push", "new"],
    "update": ["edit", "modify", "patch", "put", "change", "save", "upsert"],
    "edit": ["update", "modify", "patch", "change"],
    "save": ["store", "persist", "write", "commit", "insert", "upsert"],
    "load": ["read", "fetch", "get", "open", "import", "parse"],
    "read": ["load", "fetch", "get", "open", "parse"],
    "fetch": ["get", "load", "retrieve", "request", "query", "download"],
    "cache": ["memo", "memoize", "lru", "redis", "ttl"],
    "log": ["logger", "logging", "trace", "audit"],
    "logging": ["log", "logger", "audit"],
    "upload": ["file", "multipart", "attachment", "ingest", "import"],
    "download": ["export", "fetch", "attachment", "stream"],
    "payment": ["billing", "charge", "checkout", "invoice", "stripe", "paypal", "transaction"],
    "billing": ["payment", "invoice", "charge", "subscription"],
    "session": ["cookie", "token", "jwt", "login", "state"],
    "token": ["jwt", "bearer", "session", "credential", "secret"],
    "dashboard": ["home", "overview", "page", "view", "panel"],
    "page": ["view", "component", "screen", "template", "route"],
    "component": ["widget", "view", "page", "element"],
    "button": ["onclick", "click", "handler", "submit"],
    "schedule": ["cron", "job", "task", "worker", "queue", "timer"],
    "job": ["task", "worker", "queue", "schedule", "cron"],
    "permission": ["role", "acl", "authorize", "rbac", "policy", "access"],
    "role": ["permission", "acl", "rbac", "policy", "group"],
    "search": ["find", "query", "lookup", "filter", "index"],
    "validate": ["validation", "check", "verify", "sanitize", "schema", "assert"],
    "validation": ["validate", "check", "verify", "sanitize", "schema"],
    "parse": ["parser", "decode", "deserialize", "read", "extract"],
    "serialize": ["encode", "dump", "json", "marshal", "stringify"],
    "encrypt": ["encryption", "cipher", "crypto", "aes", "hash", "decrypt"],
    "decrypt": ["decryption", "cipher", "crypto", "encrypt"],
    "encryption": ["encrypt", "cipher", "crypto", "aes", "decrypt"],
    "message": ["msg", "chat", "text", "payload", "event"],
    "chat": ["message", "conversation", "stream", "thread"],
    "stream": ["sse", "websocket", "chunk", "generator", "yield"],
    "websocket": ["socket", "ws", "stream", "realtime"],
    "theme": ["style", "css", "color", "dark", "palette"],
    "store": ["state", "redux", "context", "persist", "save"],
    "state": ["store", "reducer", "context", "hook", "usestate"],
    "middleware": ["interceptor", "filter", "hook", "plugin"],
    "queue": ["worker", "job", "celery", "task", "broker"],
    "migration": ["migrate", "schema", "alembic", "flyway"],
    "docker": ["container", "image", "compose", "dockerfile"],
    "deploy": ["deployment", "release", "ci", "pipeline", "workflow"],
    "ci": ["pipeline", "workflow", "github", "actions", "build"],
}

# words that suggest the user is asking about a particular kind of code location
INTENT_WORDS = {
    "startup": {"start", "starts", "startup", "boot", "bootstrap", "launch", "launches",
                "begin", "begins", "initialize", "initialise", "init", "entry", "main"},
    "route": {"route", "routes", "endpoint", "endpoints", "api", "url", "http", "request",
              "requests", "controller", "handler", "rest"},
    "tests": {"test", "tests", "testing", "spec", "specs", "coverage", "pytest", "junit"},
    "callers": {"callers", "caller", "calls", "called", "uses", "used", "usages", "usage",
                "references", "invokes", "invoked"},
    "flow": {"flow", "flows", "trace", "traces", "pipeline", "lifecycle", "sequence",
             "journey", "connects", "connect", "connected", "chain"},
}


@functools.lru_cache(maxsize=100_000)
def _split_cached(text):
    spaced = _CAMEL_2.sub(r"\1 \2", _CAMEL_1.sub(r"\1 \2", text))
    return tuple(w.lower() for w in _NON_ALNUM.split(spaced) if w)


def split_identifier(text):
    """'getUserProfile' / 'get_user_profile' / 'HTTPServer' -> ['get','user','profile']."""
    if not text:
        return []
    return list(_split_cached(text))


def tokens_of(text):
    """Lower-case word tokens of a line of code or prose, with identifiers split."""
    out = []
    for ident in IDENTIFIER.findall(text or ""):
        out.extend(_split_cached(ident))
    return out


def identifier_variants(term):
    """Spellings a single concept can take in code: login user -> login_user, loginuser ..."""
    words = split_identifier(term)
    if not words:
        return []
    variants = {term.lower().strip()}
    if len(words) > 1:
        variants.update({"_".join(words), "".join(words), "-".join(words), " ".join(words)})
    return sorted(v for v in variants if v)


def _stems(word):
    stems = []
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            stems.append(stem)
            if len(stem) >= 3 and stem[-1] == stem[-2]:      # logging -> logg -> log
                stems.append(stem[:-1])
    return stems


def synonyms_for(word):
    """Code-vocabulary alternatives for one plain word (lower-case, deduplicated).

    A crude stemmer lets 'logging', 'logged' and 'logins' find the entries for 'log'/'login'.
    """
    word = word.lower()
    out = list(SYNONYMS.get(word, []))
    for stem in _stems(word):
        if stem in SYNONYMS:
            out.append(stem)
            out.extend(SYNONYMS[stem])
    seen, ordered = {word}, []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def content_words(text):
    """Words of a question that are worth searching for, in order, without duplicates."""
    words = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text or ""):
        word = raw.lower()
        if len(word) < 3 or word in STOPWORDS or word in QUESTION_NOISE:
            continue
        if word not in words:
            words.append(word)
    return words
