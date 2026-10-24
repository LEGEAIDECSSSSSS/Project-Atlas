import test from "node:test";
import assert from "node:assert/strict";
import { parseInfo, normalizeFences, extractCodeBlocks, blocksToFiles, pathFromFirstLine } from "../src/lib/markdown.js";

test("parseInfo handles the common ways models write paths", () => {
  assert.deepEqual(parseInfo("python src/utils/helpers.py"), { lang: "python", path: "src/utils/helpers.py" });
  assert.deepEqual(parseInfo("python:src/app.py"), { lang: "python", path: "src/app.py" });
  assert.deepEqual(parseInfo("src/app.py"), { lang: "python", path: "src/app.py" });
  assert.deepEqual(parseInfo('js title="web/index.js"'), { lang: "js", path: "web/index.js" });
  assert.deepEqual(parseInfo("typescript path=src/a.ts"), { lang: "typescript", path: "src/a.ts" });
  assert.deepEqual(parseInfo("python"), { lang: "python", path: "" });
  assert.deepEqual(parseInfo(""), { lang: "", path: "" });
  assert.deepEqual(parseInfo("c++"), { lang: "c++", path: "" });
  assert.deepEqual(parseInfo("Dockerfile"), { lang: "dockerfile", path: "Dockerfile" });
  assert.deepEqual(parseInfo("python main.py extra"), { lang: "python", path: "main.py" });
});

test("normalizeFences only rewrites opening fences", () => {
  const md = "text\n```python:src/a.py\nx = 1\n```\n\n```src/b.js\ny\n```\n";
  assert.equal(normalizeFences(md), "text\n```python src/a.py\nx = 1\n```\n\n```javascript src/b.js\ny\n```\n");
});

test("inline triple backticks are not fences", () => {
  const md = "use ```x``` here\n```py\na\n```";
  assert.equal(extractCodeBlocks(md).length, 1);
});

test("extractCodeBlocks: paths, nested fences, unterminated streaming block", () => {
  const md = [
    "Fix:", "```python src/a.py", "def f():", "    return 1", "```",
    "and", "````md", "```inner", "```", "````",
    "```java", "// file: com/x/Main.java", "class Main {}", "```",
    "```python src/streaming.py", "partial = ",
  ].join("\n");
  const blocks = extractCodeBlocks(md);
  assert.equal(blocks.length, 3);
  assert.equal(blocks[0].path, "src/a.py");
  assert.equal(blocks[1].lang, "md");
  assert.ok(blocks[1].code.includes("```inner"));
  assert.equal(blocks[2].path, "com/x/Main.java");
});

test("blocksToFiles names snippets and dedupes paths", () => {
  const files = blocksToFiles([
    { lang: "python", path: "a.py", code: "1" },
    { lang: "python", path: "a.py", code: "2" },
    { lang: "bash", path: "", code: "ls" },
    { lang: "bash", path: "", code: "   " },
  ]);
  assert.deepEqual(files.map((f) => f.path), ["a.py", "a-2.py", "snippet-3.sh"]);
  assert.equal(files[2].hasPath, false);
});

test("pathFromFirstLine", () => {
  assert.equal(pathFromFirstLine("# file: src/x.py\nprint()"), "src/x.py");
  assert.equal(pathFromFirstLine("<!-- path: index.html -->"), "index.html");
  assert.equal(pathFromFirstLine("print('file: no')"), "");
});
