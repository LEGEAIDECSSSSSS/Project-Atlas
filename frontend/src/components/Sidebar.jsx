import Logo from "./Logo";
import { KICKER, WORDMARK } from "../brand";
import { IconTrash } from "./Icons";

export default function Sidebar({
  conversations, activeId, onSelect, onNew, onDelete, onRename, open, onClose,
  health, theme, onToggleTheme,
}) {
  const status =
    health.state === "loading" ? { cls: "wait", text: "Connecting" }
    : health.state === "down" ? { cls: "bad", text: "Backend offline" }
    : !health.ollama ? { cls: "bad", text: "Ollama offline" }
    : { cls: "ok", text: "On this machine" };

  return (
    <>
      <div className={`scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <Logo size={36} />
          <div>
            <span className="wordmark">{WORDMARK}</span>
            <span className="kicker">{KICKER}</span>
          </div>
        </div>

        <button type="button" className="new-chat" onClick={onNew}>
          <span aria-hidden="true">+</span> New conversation
        </button>

        <div className="rail-label">
          <span>Log</span>
          <span>{String(conversations.length).padStart(2, "0")}</span>
        </div>

        <nav className="history" aria-label="Conversations">
          {conversations.length === 0 && <p className="empty-history">Nothing here yet. Ask something.</p>}
          {conversations.map((c) => (
            <div key={c.id} className={`history-item ${c.id === activeId ? "active" : ""}`}>
              <button
                type="button"
                className="history-title"
                onClick={() => onSelect(c.id)}
                onDoubleClick={() => onRename(c)}
                title={`${c.title} (double-click to rename)`}
              >
                {c.title}
              </button>
              <button type="button" className="history-delete" onClick={() => onDelete(c)} aria-label={`Delete ${c.title}`}>
                <IconTrash width={14} height={14} />
              </button>
            </div>
          ))}
        </nav>

        <div className="sidebar-foot">
          <span className={`status ${status.cls}`} title={health.error || ""}>
            <i /> {status.text}
          </span>
                    <div className="theme-switch" role="group" aria-label="Theme">
            <button type="button" className={theme === "dark" ? "on" : ""} aria-pressed={theme === "dark"} onClick={() => theme !== "dark" && onToggleTheme()}>
              Dark
            </button>
            <button type="button" className={theme === "light" ? "on" : ""} aria-pressed={theme === "light"} onClick={() => theme !== "light" && onToggleTheme()}>
              Light
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}