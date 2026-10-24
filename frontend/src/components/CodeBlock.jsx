import { useState } from "react";
import { copyText, saveText } from "../lib/download";
import { fileNameFor } from "../lib/markdown";
import { IconCheck, IconCopy, IconDownload, IconFile } from "./Icons";

export default function CodeBlock({ lang, path, code, children }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    await copyText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="codeblock">
      <div className="codeblock-bar">
        <span className="codeblock-name" title={path || lang}>
          {path ? <><IconFile width={14} height={14} /> {path}</> : lang || "code"}
        </span>
        <span className="codeblock-actions">
          <button type="button" onClick={copy} aria-label="Copy code">
            {copied ? <IconCheck /> : <IconCopy />} {copied ? "Copied" : "Copy"}
          </button>
          <button type="button" onClick={() => saveText(code + "\n", fileNameFor({ lang, path }))} aria-label="Download file">
            <IconDownload /> Download
          </button>
        </span>
      </div>
      <pre><code className={lang ? `hljs language-${lang}` : "hljs"}>{children}</code></pre>
    </div>
  );
}
