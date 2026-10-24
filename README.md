# Local Coder

A private AI assistant that runs entirely on your laptop through [Ollama](https://ollama.com):
a **coding assistant** (paste code, attach a ZIP, get fixes and downloadable files) and an
**everyday chat assistant** (summaries, lists and totals, writing, questions).

## Run it

1. **Ollama** must be running. Pull the two models once (about 4.7 GB each):
   ```
   ollama pull qwen2.5-coder:7b
   ollama pull qwen2.5:7b
   ```
   If the chat model is missing, everything falls back to the coding model and the UI tells you.
2. **Backend** (from `backend/`, with your venv active):
   ```
   pip install -r requirements.txt
   uvicorn main:app --port 8000 --reload
   ```
3. **Frontend** (from `frontend/`): `npm install` then `npm run dev`, open http://localhost:5173.

On Windows, `start.bat` in this folder does steps 2 and 3 and opens the browser.

## Using it

- **Auto / Chat / Code** (top bar): Auto sends coding questions to the code model and everything
  else to the chat model. Shown on every reply, so you can see what it chose.
- **Paste code** straight into the box, or click the paperclip / drag and drop:
  a **.zip** project, or loose code files. Big junk folders (`node_modules`, `.venv`, `.git`,
  `build`, `target`...) are skipped automatically.
- **Getting files back**: when a reply contains code, each block has Copy and Download. If the model
  names the file path, the reply gets **Download .zip** and, with a project attached,
  **Apply to project**. Apply shows what it will replace, backs up the old files
  (`data/backups/`), and keeps Windows line endings. The paperclip chip downloads the whole
  updated project as a ZIP.
- **Lists and totals**: the model uses an exact calculator tool instead of doing arithmetic itself.
- **Stop** any reply with the stop button or Esc. Chats are saved (SQLite, `data/`) and survive restarts.
  Double-click a chat in the sidebar to rename it.

## Tuning for a 6 GB GTX 1660 Ti

Defaults: 8192-token context, 3072-token answers, model kept loaded for 10 minutes. Run `ollama ps`
while a chat is open: it should say **100% GPU**. If it shows a CPU split, set a smaller context
before starting the backend (Windows: `set LC_NUM_CTX=4096`). Only one 7B model fits in VRAM at a
time, so the first message after switching between Chat and Code takes a few extra seconds while
the model swaps. To avoid swaps, set both to the same model:
`set LC_CHAT_MODEL=qwen2.5-coder:7b`.

| Variable | Default | Meaning |
|---|---|---|
| `LC_CODE_MODEL` / `LC_CHAT_MODEL` | `qwen2.5-coder:7b` / `qwen2.5:7b` | Models per mode |
| `LC_NUM_CTX` | 8192 | Context window (lower = less VRAM) |
| `LC_NUM_PREDICT` | 3072 | Max tokens per answer |
| `LC_KEEP_ALIVE` | 10m | How long a model stays in memory |
| `LC_APP_NAME` | Local Coder | Name used in the chat prompt |
| `LC_DATA_DIR` | `../data` | Chats, uploaded projects, backups |

## Renaming and logo

Frontend: `frontend/src/brand.js` (name), `frontend/public/logo.svg` and `favicon.svg` (the current
logo is a placeholder), `<title>` in `frontend/index.html`. Backend: `LC_APP_NAME`.

## How it works

- `backend/agent/agent.py`: the agent loop. The model asks for tools with a JSON message
  (`search_project`, `read_project_file`, `list_project_files`, `run_project_tests`, `calculate`),
  results are fed back, and the final answer streams to the browser. Tool JSON is never shown.
- `agent/router.py` picks Chat vs Code; `agent/context.py` trims history to fit the context window.
- When a project is attached, the model is given its file list, and the full contents if it is small.
- `services/` holds Ollama streaming, the SQLite store, safe ZIP handling, and the calculator.
- `backend/tests/` (`pytest`) and `frontend/tests/` (`npm test`).

## Known limits and ideas for next

- A 7B model is a 7B model: it is decent for everyday tasks and focused code fixes, but it will not
  match a large cloud model on big refactors. Keep requests focused.
- "Apply to project" replaces whole files (only blocks the model labelled with a path). Partial
  edits via search/replace patches would be a good next step for large files.
- `run_project_tests` still supports Maven only. Adding `pytest` / `npm test` is straightforward.
- No web access, no image input. A very large pasted file is cut to fit the context (you get a warning);
  upload it as a ZIP or file instead so the assistant can read it in pieces.
