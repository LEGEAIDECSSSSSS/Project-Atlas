// Helpers for finding code blocks (and the file paths the model attaches to them).

const LANG_BY_EXT = {
  py: "python", js: "javascript", jsx: "javascript", mjs: "javascript", ts: "typescript",
  tsx: "typescript", java: "java", c: "c", h: "c", cpp: "cpp", hpp: "cpp", cs: "csharp",
  go: "go", rs: "rust", rb: "ruby", php: "php", html: "xml", xml: "xml", vue: "xml",
  svelte: "xml", css: "css", scss: "scss", json: "json", yml: "yaml", yaml: "yaml",
  sh: "bash", bash: "bash", md: "markdown", sql: "sql", kt: "kotlin", kts: "kotlin",
  swift: "swift", toml: "ini", ini: "ini", properties: "ini", bat: "dos", ps1: "powershell",
  gradle: "groovy", txt: "plaintext",
};

const EXT_BY_LANG = {
  python: "py", py: "py", javascript: "js", js: "js", jsx: "jsx", typescript: "ts", ts: "ts",
  tsx: "tsx", java: "java", c: "c", cpp: "cpp", "c++": "cpp", csharp: "cs", "c#": "cs",
  go: "go", rust: "rs", ruby: "rb", php: "php", html: "html", xml: "xml", css: "css",
  scss: "scss", json: "json", yaml: "yml", yml: "yml", bash: "sh", sh: "sh", shell: "sh",
  zsh: "sh", markdown: "md", md: "md", sql: "sql", kotlin: "kt", swift: "swift", ini: "ini",
  toml: "toml", powershell: "ps1", dos: "bat", batch: "bat", plaintext: "txt", text: "txt",
};

const PATH_KEY = /^(?:title|path|file|filename|name)=/i;
const SPECIAL_NAMES = /^(Dockerfile|Makefile|Procfile|Gemfile|Rakefile|\.gitignore|\.dockerignore|\.env[\w.]*)$/i;
const FENCE = /^(\s{0,3})(`{3,})(.*)$/;

function looksLikePath(s) {
  if (!s || !/^[\w@.\-/\\]+$/.test(s)) return false;
  const last = s.split(/[\\/]/).pop();
  return /\.[A-Za-z0-9]{1,10}$/.test(last) || SPECIAL_NAMES.test(last);
}

function cleanToken(t) {
  return t.replace(PATH_KEY, "").replace(/^["'`]+|["'`,;]+$/g, "");
}

export function languageFromPath(path) {
  const last = (path || "").split(/[\\/]/).pop();
  if (/^Dockerfile$/i.test(last)) return "dockerfile";
  if (/^Makefile$/i.test(last)) return "makefile";
  const ext = last.includes(".") ? last.split(".").pop().toLowerCase() : "";
  return LANG_BY_EXT[ext] || "";
}

export function extensionForLanguage(lang) {
  return EXT_BY_LANG[(lang || "").toLowerCase()] || "txt";
}

/** Split a fence info string ("python src/a.py", "python:src/a.py", "src/a.py") into lang + path. */
export function parseInfo(info) {
  let lang = "";
  let path = "";
  const tokens = (info || "").trim().split(/\s+/).filter(Boolean);

  tokens.forEach((raw, i) => {
    const colon = raw.match(/^([A-Za-z][\w+#-]*):(?!\/\/)(.+)$/);
    if (colon && !path && looksLikePath(cleanToken(colon[2]))) {
      if (!lang) lang = colon[1];
      path = cleanToken(colon[2]);
      return;
    }
    const candidate = cleanToken(raw);
    if (!path && looksLikePath(candidate)) {
      path = candidate;
    } else if (i === 0 && !lang) {
      lang = raw.replace(/^[{.]+|[}]+$/g, "");
    }
  });

  // First-line "file:" comments are handled separately (see pathFromFirstLine).
  if (path && !lang) lang = languageFromPath(path);
  return { lang, path: path.replace(/\\/g, "/").replace(/^\.\//, "") };
}

/** Models sometimes put the path in a first-line comment: "# file: src/a.py". */
export function pathFromFirstLine(code) {
  const first = (code || "").split("\n", 1)[0];
  const m = first.match(/^\s*(?:#|\/\/|--|\/\*|<!--)\s*(?:file(?:name)?|path)\s*:\s*([^\s*>]+)/i);
  return m && looksLikePath(m[1]) ? m[1].replace(/\\/g, "/").replace(/^\.\//, "") : "";
}

/** Rewrite opening fences so the language is always the first word and the path the second. */
export function normalizeFences(text) {
  const lines = text.split("\n");
  let open = 0;

  for (let i = 0; i < lines.length; i++) {
    const m = FENCE.exec(lines[i]);
    if (!m) continue;
    const info = m[3].trim();

    if (!open) {
      if (info.includes("`")) continue; // inline code such as ```x```, not a fence
      const { lang, path } = parseInfo(info);
      lines[i] = `${m[1]}${m[2]}${[lang, path].filter(Boolean).join(" ")}`;
      open = m[2].length;
    } else if (m[2].length >= open && info === "") {
      open = 0;
    }
  }
  return lines.join("\n");
}

/** All fenced code blocks in a markdown string: [{ lang, path, code }]. */
export function extractCodeBlocks(text) {
  const lines = normalizeFences(text || "").split("\n");
  const blocks = [];
  let current = null;

  for (const line of lines) {
    const m = FENCE.exec(line);
    if (!current) {
      if (m && !m[3].includes("`")) {
        const { lang, path } = parseInfo(m[3]);
        current = { lang, path, fence: m[2].length, body: [] };
      }
    } else if (m && m[2].length >= current.fence && m[3].trim() === "") {
      const code = current.body.join("\n");
      blocks.push({ lang: current.lang, path: current.path || pathFromFirstLine(code), code });
      current = null;
    } else {
      current.body.push(line);
    }
  }
  return blocks; // an unterminated block (still streaming) is intentionally ignored
}

/** Files to offer for download: every block, with a fallback name when the model gave no path. */
export function blocksToFiles(blocks) {
  const used = new Set();
  return blocks
    .filter((b) => b.code.trim())
    .map((b, i) => {
      let path = b.path || `snippet-${i + 1}.${extensionForLanguage(b.lang)}`;
      while (used.has(path)) path = path.replace(/(\.[^./]+)?$/, `-${i + 1}$1`);
      used.add(path);
      return { path, content: b.code.endsWith("\n") ? b.code : b.code + "\n", hasPath: Boolean(b.path) };
    });
}

export function fileNameFor(block) {
  if (block.path) return block.path.split("/").pop();
  return `snippet.${extensionForLanguage(block.lang)}`;
}
