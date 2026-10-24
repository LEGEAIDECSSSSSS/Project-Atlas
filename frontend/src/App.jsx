import { useCallback, useEffect, useRef, useState } from "react";
import {
  applyFiles, deleteConversation, getConversation, getHealth, listConversations,
  projectZipUrl, renameConversation, streamChat, uploadFiles, uploadProject, zipFiles,
} from "./api";
import { APP_NAME } from "./brand";
import Composer from "./components/Composer";
import { IconMenu } from "./components/Icons";
import Message from "./components/Message";
import Sidebar from "./components/Sidebar";
import { saveBlob, slug } from "./lib/download";
import { useLocalStorage } from "./lib/hooks";
import Welcome from "./components/Welcome";
const newId = () => crypto.randomUUID();

const MODES = [
  { id: "auto", label: "Auto", hint: "Picks Chat or Code for you" },
  { id: "chat", label: "Chat", hint: "Everyday tasks, writing, summaries, maths" },
  { id: "code", label: "Code", hint: "Programming and attached projects" },
];

function fromServer(m) {
  return { id: `m-${m.id}`, role: m.role, content: m.content, meta: m.meta || {} };
}

export default function App() {
  const [activeId, setActiveId] = useState(newId);
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [project, setProject] = useState(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [health, setHealth] = useState({ state: "loading" });
  const [toasts, setToasts] = useState([]);

  const [mode, setMode] = useLocalStorage("lc.mode", "auto");
  const [model, setModel] = useLocalStorage("lc.model", "");
  const [theme, setTheme] = useLocalStorage("lc.theme", "dark");

  const abortRef = useRef(null);
  const scrollRef = useRef(null);
  const stickRef = useRef(true);
  const inputRef = useRef(null);

  // ---- helpers -----------------------------------------------------------
  const toast = useCallback((text, kind = "info") => {
    const id = newId();
    setToasts((t) => [...t, { id, text, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 7000 : 4000);
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations((await listConversations()).conversations);
    } catch {
      /* the health banner already explains this */
    }
  }, []);

  // ---- effects -----------------------------------------------------------
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const h = await getHealth();
        if (alive) setHealth({ state: "up", ...h });
      } catch (error) {
        if (alive) setHealth({ state: "down", error: error.message });
      }
    };
    check();
    listConversations()
      .then((r) => alive && setConversations(r.conversations))
      .catch(() => { });
    const timer = setInterval(check, 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && messages.length > 0 && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // ---- conversations -----------------------------------------------------
  function resetView(id) {
    setActiveId(id);
    setMessages([]);
    setProject(null);
    setInput("");
    setSidebarOpen(false);
    stickRef.current = true;
  }

  function startNew() {
    if (busy) return;
    resetView(newId());
    inputRef.current?.focus();
  }

  async function selectConversation(id) {
    if (busy || id === activeId) return setSidebarOpen(false);
    try {
      const c = await getConversation(id);
      resetView(id);
      setMessages(c.messages.map(fromServer));
      setProject(c.project_id ? { id: c.project_id, name: c.project_name || "project" } : null);
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function removeConversation(c) {
    if (!window.confirm(`Delete "${c.title}"? Any project attached to it is deleted too.`)) return;
    try {
      await deleteConversation(c.id);
      if (c.id === activeId) resetView(newId());
      refreshConversations();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function renameChat(c) {
    const title = window.prompt("Rename chat", c.title);
    if (!title || !title.trim()) return;
    try {
      await renameConversation(c.id, title.trim());
      refreshConversations();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  // ---- sending -----------------------------------------------------------
  async function send({ regenerate = false, resendText = null } = {}) {
    if (busy) return;
    const text = (resendText ?? input).trim();
    if (!regenerate && !text) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    stickRef.current = true;

    const assistant = {
      id: `a-${newId()}`, role: "assistant", content: "", streaming: true,
      status: "Thinking…", meta: {}, warnings: [],
    };
    const patch = (changes) =>
      setMessages((all) => all.map((m) => (m.id === assistant.id ? { ...m, ...changes } : m)));

    setMessages((all) => {
      if (regenerate) {
        const trimmed = all[all.length - 1]?.role === "assistant" ? all.slice(0, -1) : all;
        return [...trimmed, assistant];
      }
      return [...all, { id: `u-${newId()}`, role: "user", content: text }, assistant];
    });
    if (!regenerate) setInput("");

    let acc = "";
    let frame = 0;
    const tools = [];
    let meta = {};
    let reachedServer = false;
    const flush = () => {
      frame = 0;
      patch({ content: acc });
    };

    try {
      const stream = streamChat(
        {
          message: regenerate ? "" : text,
          conversation_id: activeId,
          regenerate,
          mode,
          model: model || null,
          project_id: project?.id ?? "",
          project_name: project?.name ?? null,
        },
        controller.signal,
      );

      for await (const event of stream) {
        reachedServer = true;
        switch (event.type) {
          case "meta":
            meta = { mode: event.mode, model: event.model };
            patch({ meta });
            break;
          case "status":
            patch({ status: event.text });
            break;
          case "token":
            acc += event.text;
            if (!frame) frame = requestAnimationFrame(flush);
            break;
          case "tool":
            tools.push(event.label);
            patch({ meta: { ...meta, tools: [...tools] } });
            break;
          case "warning":
            setMessages((all) =>
              all.map((m) => (m.id === assistant.id ? { ...m, warnings: [...m.warnings, event.text] } : m)),
            );
            break;
          case "error":
            patch({ error: event.message, retry: "regenerate" });
            break;
          case "done":
            patch({ meta: event.meta });
            break;
          default:
        }
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        patch({ error: error.message, retry: reachedServer ? "regenerate" : "resend", resendText: text });
      }
    } finally {
      if (frame) cancelAnimationFrame(frame);
      patch({ content: acc, streaming: false, status: "" });
      abortRef.current = null;
      setBusy(false);
      refreshConversations();
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function retry(message) {
    if (message.retry === "resend") send({ resendText: message.resendText });
    else send({ regenerate: true });
  }

  // ---- files -------------------------------------------------------------
  const handleFiles = useCallback(
    async (fileList) => {
      const files = Array.from(fileList || []);
      if (!files.length) return;

      setUploading(true);
      try {
        const isZip = (f) => /\.zip$/i.test(f.name);
        const zips = files.filter(isZip);
        const loose = files.filter((f) => !isZip(f));
        let current = project;

        if (zips.length) {
          const r = await uploadProject(zips[0]);
          current = { id: r.project_id, name: r.name, files: r.files };
          if (r.skipped) toast(`Skipped ${r.skipped} generated/huge files (node_modules, .venv, build output…).`);
          if (zips.length > 1) toast("Only the first ZIP was attached.");
        }
        if (loose.length) {
          const r = await uploadFiles(loose, current?.id);
          current = {
            id: r.project_id,
            name: current?.name || r.name,
            files: (current?.files || 0) + r.files,
          };
        }
        setProject(current);
        toast(`Attached ${current.name}. Ask me anything about it.`);
        inputRef.current?.focus();
      } catch (error) {
        toast(error.message, "error");
      } finally {
        setUploading(false);
      }
    },
    [project, toast],
  );

  async function downloadZip(files) {
    try {
      const blob = await zipFiles(files.map(({ path, content }) => ({ path, content })), "files");
      saveBlob(blob, `${slug(APP_NAME)}-files.zip`);
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function applyToProject(files) {
    if (!project) return;
    const payload = files.map(({ path, content }) => ({ path, content }));
    try {
      const plan = await applyFiles(project.id, payload, true);
      const lines = plan.files.map((f) =>
        f.status === "created"
          ? `• ${f.path} — new file`
          : `• ${f.path} — replaces existing (${f.old_lines} → ${f.new_lines} lines)${f.shrunk ? "  ⚠ much shorter than before" : ""}`,
      );
      const ok = window.confirm(
        `Apply ${files.length} file${files.length === 1 ? "" : "s"} to the project?\n\n${lines.join("\n")}\n\nReplaced files are backed up first.`,
      );
      if (!ok) return;
      await applyFiles(project.id, payload, false);
      toast("Applied. Use the download icon on the project chip to get the updated ZIP.");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  // ---- drag & drop -------------------------------------------------------
  const dragProps = {
    onDragOver: (e) => {
      if (e.dataTransfer.types.includes("Files")) {
        e.preventDefault();
        setDragging(true);
      }
    },
    onDragLeave: (e) => {
      if (!e.relatedTarget) setDragging(false);
    },
    onDrop: (e) => {
      e.preventDefault();
      setDragging(false);
      handleFiles(e.dataTransfer.files);
    },
  };

  // ---- render ------------------------------------------------------------
  const lastAssistantId = [...messages].reverse().find((m) => m.role === "assistant")?.id;
  const installed = health.models || [];
  const chatModelMissing = health.state === "up" && health.ollama && !installed.includes(health.chat_model);
  const activeTitle = conversations.find((c) => c.id === activeId)?.title;

  return (
    <div className="app" {...dragProps}>
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={selectConversation}
        onNew={startNew}
        onDelete={removeConversation}
        onRename={renameChat}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        health={health}
        theme={theme}
        onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
      />

      <main className="main">
        <header className="topbar">
          <button type="button" className="icon-btn menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <IconMenu />
          </button>
          <div className="topbar-title">{activeTitle || "New chat"}</div>

          <div className="segmented" role="group" aria-label="Mode">
            {MODES.map((m) => (
              <button
                key={m.id}
                type="button"
                className={mode === m.id ? "on" : ""}
                onClick={() => setMode(m.id)}
                title={m.hint}
                aria-pressed={mode === m.id}
              >
                {m.label}
              </button>
            ))}
          </div>

          <select
            className="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            aria-label="Model"
            title="Model to use"
          >
            <option value="">Model: automatic</option>
            {installed.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </header>

        {health.state === "down" && (
          <div className="banner bad">
            Can’t reach the backend at port 8000. Start it with <code>uvicorn main:app --port 8000</code> (see README).
          </div>
        )}
        {health.state === "up" && !health.ollama && (
          <div className="banner bad">Ollama isn’t running. Open the Ollama app (or run <code>ollama serve</code>).</div>
        )}
        {chatModelMissing && (
          <div className="banner">
            For much better everyday chat, run <code>ollama pull {health.chat_model}</code>. Until then
            I’ll use the coding model for everything.
          </div>
        )}

        <div
          className="scroll"
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
        >
          {messages.length === 0 ? (
            <Welcome
              onPick={(text) => {
                setInput(text);
                inputRef.current?.focus();
              }}
            />
          ) : (
            <div className="thread">
              {messages.map((m) => (
                <Message
                  key={m.id}
                  message={m}
                  isLast={m.id === lastAssistantId}
                  busy={busy}
                  project={project}
                  onRegenerate={retry}
                  onZip={downloadZip}
                  onApply={applyToProject}
                />
              ))}
            </div>
          )}
        </div>

        <Composer
          value={input}
          onChange={setInput}
          onSend={() => send()}
          onStop={stop}
          busy={busy}
          uploading={uploading}
          onFiles={handleFiles}
          project={project}
          onDetach={() => setProject(null)}
          projectZipUrl={project ? projectZipUrl(project.id, project.name) : "#"}
          placeholder={project ? `Ask about ${project.name}…` : "Message, paste code, or attach a ZIP…"}
          inputRef={inputRef}
        />
      </main>

      {dragging && <div className="drop-overlay">Drop a ZIP or code files to attach</div>}

      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>{t.text}</div>
        ))}
      </div>
    </div>
  );
}
