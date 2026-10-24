export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function fail(res) {
  let detail = "";
  try {
    detail = (await res.json()).detail;
  } catch {
    /* not JSON */
  }
  throw new Error(detail || `Server returned ${res.status}`);
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, options);
  } catch (error) {
    throw new Error("Cannot reach the backend. Is it running on port 8000?", { cause: error });
  }
  if (!res.ok) await fail(res);
  return res;
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getHealth = () => request("/health").then((r) => r.json());
export const listConversations = () => request("/conversations").then((r) => r.json());
export const getConversation = (id) => request(`/conversations/${id}`).then((r) => r.json());
export const deleteConversation = (id) => request(`/conversations/${id}`, { method: "DELETE" });
export const renameConversation = (id, title) =>
  request(`/conversations/${id}`, { ...json({ title }), method: "PATCH" }).then((r) => r.json());

export function uploadProject(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/upload-project", { method: "POST", body: form }).then((r) => r.json());
}

export function uploadFiles(files, projectId) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));
  if (projectId) form.append("project_id", projectId);
  return request("/upload-files", { method: "POST", body: form }).then((r) => r.json());
}

export const applyFiles = (projectId, files, dryRun = false) =>
  request(`/projects/${projectId}/apply`, json({ files, dry_run: dryRun })).then((r) => r.json());

export const zipFiles = (files, name) =>
  request("/download-zip", json({ files, name })).then((r) => r.blob());

export const projectZipUrl = (projectId, name) =>
  `${API_URL}/projects/${projectId}/download?name=${encodeURIComponent(name || "project")}`;

/** POST /chat/stream and yield each server event as an object. */
export async function* streamChat(body, signal) {
  let res;
  try {
    res = await fetch(`${API_URL}/chat/stream`, { ...json(body), signal });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("Cannot reach the backend. Is it running on port 8000?", { cause: error });
  }
  if (!res.ok) await fail(res);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data:")) yield JSON.parse(line.slice(5).trim());
      }
    }
  }
}
