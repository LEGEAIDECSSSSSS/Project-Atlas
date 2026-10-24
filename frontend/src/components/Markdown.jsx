import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import CodeBlock from "./CodeBlock";
import { normalizeFences, parseInfo, pathFromFirstLine } from "../lib/markdown";

const remarkPlugins = [remarkGfm];
const rehypePlugins = [[rehypeHighlight, { detect: false, ignoreMissing: true }]];

function nodeText(node) {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  return nodeText(node.props?.children);
}

const components = {
  pre({ node, children }) {
    const el = Array.isArray(children) ? children[0] : children;
    const props = el?.props ?? {};
    const lang = /language-(\S+)/.exec(props.className || "")?.[1] || "";
    const code = nodeText(props.children).replace(/\n$/, "");
    const meta = node?.children?.[0]?.data?.meta || "";
    const path = (meta && parseInfo(meta).path) || pathFromFirstLine(code);
    return <CodeBlock lang={lang} path={path} code={code}>{props.children}</CodeBlock>;
  },
  a({ href, children }) {
    return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
  },
  table({ children }) {
    return <div className="table-wrap"><table>{children}</table></div>;
  },
};

function Markdown({ text }) {
  const normalized = useMemo(() => normalizeFences(text || ""), [text]);
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>
        {normalized}
      </ReactMarkdown>
    </div>
  );
}

export default memo(Markdown);
