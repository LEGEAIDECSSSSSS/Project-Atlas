import { memo, useMemo, useState } from "react";
import Markdown from "./Markdown";
import { WORDMARK } from "../brand";
import { copyText } from "../lib/download";
import { blocksToFiles, extractCodeBlocks } from "../lib/markdown";
import { IconCheck, IconCopy, IconDownload, IconFolder, IconRefresh, IconWrench } from "./Icons";

// A small compass needle that swings while Atlas is working.
function Needle() {
  return (
    <svg className="needle" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
      <g className="needle-swing">
        <path d="M12 2 15 12H9Z" fill="var(--accent)" />
        <path d="M12 22 9 12h6Z" fill="currentColor" opacity=".45" />
      </g>
      <circle cx="12" cy="12" r="1.6" fill="var(--bg)" />
    </svg>
  );
}

const MODE_LABEL = { chat: "Chat", code: "Code" };

function UserMessage({ message }) {
  const [expanded, setExpanded] = useState(false);
  const long = message.content.length > 1500;
  const text = long && !expanded ? message.content.slice(0, 1500) + "…" : message.content;

  return (
    <div className="msg msg-user">
      <div className="gutter">You</div>
      <div className="bubble">
        <div className="user-text">{text}</div>
        {long && (
          <button type="button" className="link-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? "Show less" : `Show all (${message.content.length.toLocaleString()} characters)`}
          </button>
        )}
      </div>
    </div>
  );
}

function FileBar({ files, project, onZip, onApply }) {
  const applicable = files.filter((f) => f.hasPath);
  return (
    <div className="filebar">
      <span className="filebar-label">
        {files.length} file{files.length === 1 ? "" : "s"} in this reply
      </span>
      <button type="button" onClick={() => onZip(files)}>
        <IconDownload /> Download .zip
      </button>
      {project && applicable.length > 0 && (
        <button type="button" onClick={() => onApply(applicable)} title="Replaces whole files in the attached project (a backup is kept)">
          <IconFolder /> Apply {applicable.length} to project
        </button>
      )}
    </div>
  );
}

function AssistantMessage({ message, isLast, busy, project, onRegenerate, onZip, onApply }) {
  const [copied, setCopied] = useState(false);
  const meta = message.meta || {};
  const done = !message.streaming;

  const files = useMemo(
    () => (done ? blocksToFiles(extractCodeBlocks(message.content)) : []),
    [done, message.content],
  );
  const showFiles = files.length >= 2 || files.some((f) => f.hasPath);

  async function copy() {
    await copyText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="msg msg-assistant">
      <div className="gutter accent">{WORDMARK}</div>
      <div className="msg-body">
        <div className="msg-head">
          {meta.mode && (
            <span className="tag">{MODE_LABEL[meta.mode]} · {meta.model}</span>
          )}
          {meta.stats?.tokens_per_second && done && (
            <span className="tag subtle">{meta.stats.tokens_per_second} tok/s</span>
          )}
          {meta.stopped && <span className="tag subtle">stopped</span>}
        </div>

        {message.warnings?.map((w) => (
          <div key={w} className="notice">{w}</div>
        ))}

        {meta.tools?.length > 0 && (
          <div className="chips">
            {meta.tools.map((t, i) => (
              <span key={i} className="chip"><IconWrench width={12} height={12} /> {t}</span>
            ))}
          </div>
        )}

        {message.content ? (
          <Markdown text={message.content} />
        ) : message.streaming && !message.error ? (
          <div className="thinking">
            <Needle />
            <span>{message.status || "Thinking…"}</span>
          </div>
        ) : null}

        {message.streaming && message.content && message.status && message.status !== "Thinking…" && (
          <div className="thinking small"><Needle /><span>{message.status}</span></div>
        )}

        {message.error && (
          <div className="error-box">
            <span>{message.error}</span>
            {!busy && (
              <button type="button" onClick={() => onRegenerate(message)}>
                <IconRefresh /> Try again
              </button>
            )}
          </div>
        )}

        {done && showFiles && (
          <FileBar files={files} project={project} onZip={onZip} onApply={onApply} />
        )}

        {done && message.content && (
          <div className="msg-actions">
            <button type="button" onClick={copy} aria-label="Copy reply">
              {copied ? <IconCheck /> : <IconCopy />} {copied ? "Copied" : "Copy"}
            </button>
            {isLast && !busy && (
              <button type="button" onClick={() => onRegenerate(message)} aria-label="Regenerate reply">
                <IconRefresh /> Regenerate
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Message(props) {
  return props.message.role === "user" ? <UserMessage {...props} /> : <AssistantMessage {...props} />;
}

export default memo(Message);