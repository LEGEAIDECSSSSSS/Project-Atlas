@echo off
cd /d "%~dp0"
where ollama >nul 2>&1 || echo [!] Ollama was not found on PATH. Install it from https://ollama.com and run it first.
start "Backend" cmd /k "cd backend && call .venv\Scripts\activate && uvicorn main:app --port 8000 --reload"
start "Frontend" cmd /k "cd frontend && npm run dev"
timeout /t 5 >nul
start http://localhost:5173
