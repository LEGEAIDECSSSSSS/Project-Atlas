export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function saveText(text, filename) {
  saveBlob(new Blob([text], { type: "text/plain;charset=utf-8" }), filename);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Fallback for non-secure contexts.
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

export const slug = (s) =>
  (s || "files").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "files";
