import Logo from "./Logo";
import { APP_NAME } from "../brand";
import { IconPlus, IconSun, IconMoon, IconTrash } from "./Icons";

export default function Sidebar({
  conversations, activeId, onSelect, onNew, onDelete, onRename, open, onClose,
  health, theme, onToggleTheme,
}) {
  const status =
    health.state === "loading" ? { cls: "wait", text: "Connecting…" }
    : health.state === "down" ? { cls: "bad", text: "Backend offline" }
    : !health.ollama ? { cls: "bad", text: "Ollama not running" }
    : { cls: "ok", text: "Ollama connected" };

  return (
    <>
      <div className={`scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <Logo size={30} />
          <span>{APP_NAME}</span>
        </div>

        <button type="button" className="new-chat" onClick={onNew}>
          <IconPlus /> New chat
        </button>

        <nav className="history" aria-label="Conversations">
          {conversations.length === 0 && <p className="empty-history">Your chats will appear here.</p>}
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
          <button type="button" className="icon-btn" onClick={onToggleTheme} aria-label="Toggle light/dark theme">
            {theme === "dark" ? <IconSun /> : <IconMoon />}
          </button>
        </div>
      </aside>
    </>
  );
}
