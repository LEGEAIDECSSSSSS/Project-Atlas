import { useEffect, useRef } from "react";
import { IconClip, IconDownload, IconFolder, IconSend, IconStop, IconX } from "./Icons";

export default function Composer({
  value, onChange, onSend, onStop, busy, uploading, onFiles, project, onDetach,
  projectZipUrl, placeholder, inputRef,
}) {
  const fileInput = useRef(null);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [value, inputRef]);

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (!busy) onSend();
    } else if (e.key === "Escape" && busy) {
      onStop();
    }
  }

  const canSend = value.trim().length > 0 && !busy && !uploading;

  return (
    <div className="composer-wrap">
      <div className="composer">
        {(project || uploading) && (
          <div className="attachments">
            {uploading && <span className="chip attach">Uploading…</span>}
            {project && (
              <span className="chip attach">
                <IconFolder width={14} height={14} />
                <span className="attach-name">{project.name}</span>
                {project.files ? <span className="subtle">{project.files} files</span> : null}
                <a href={projectZipUrl} download title="Download the project as a ZIP (includes applied changes)">
                  <IconDownload width={14} height={14} />
                </a>
                <button type="button" onClick={onDetach} aria-label="Detach project" title="Detach project">
                  <IconX width={14} height={14} />
                </button>
              </span>
            )}
          </div>
        )}

        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          rows={1}
          aria-label="Message"
        />

        <div className="composer-row">
          <input
            ref={fileInput}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            className="icon-btn"
            onClick={() => fileInput.current.click()}
            disabled={uploading}
            aria-label="Attach a ZIP or files"
            title="Attach a ZIP project or code files (or drag & drop)"
          >
            <IconClip />
          </button>
          <span className="hint">Enter to send · Shift+Enter for a new line</span>
          {busy ? (
            <button type="button" className="send stop" onClick={onStop} aria-label="Stop generating" title="Stop (Esc)">
              <IconStop />
            </button>
          ) : (
            <button type="button" className="send" onClick={onSend} disabled={!canSend} aria-label="Send message">
              <IconSend />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
