import asyncio
import json
import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
from agent.agent import run_agent_stream
from agent.router import resolve_mode
from services import db
from services.listparse import try_total_list
from services.ollama import OllamaError, loaded_models, resolve_model
from services.project_service import build_project_context, project_path


router = APIRouter()

MAX_HISTORY_MESSAGES = 40   # loaded from the database; the agent trims further to fit
CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class ChatRequest(BaseModel):
    message: str = ""
    conversation_id: str
    project_id: str | None = None      # "" detaches the current project
    project_name: str | None = None
    mode: Literal["auto", "chat", "code"] = "auto"
    model: str | None = None
    regenerate: bool = False


def _title_from(message: str) -> str:
    title = " ".join(message.split())
    return (title[:57] + "...") if len(title) > 60 else (title or "New chat")


async def _run_chat(request: ChatRequest):
    """Shared by /chat and /chat/stream. Yields UI events as dicts."""
    conversation_id = request.conversation_id.strip()
    conversation = db.get_or_create_conversation(conversation_id)
    message = request.message

    if request.regenerate:
        previous = db.pop_last_turn(conversation_id)
        if previous is None:
            yield {"type": "error", "message": "There is nothing to regenerate yet."}
            return
        message = previous

    if not message.strip():
        yield {"type": "error", "message": "Please type a message first."}
        return

    # Save the user's message first, so "Try again" always has something to retry,
    # even if the model or Ollama turns out to be unavailable.
    history = db.get_messages(conversation_id, limit=MAX_HISTORY_MESSAGES)
    is_first = not history

    db.add_message(conversation_id, "user", message)
    if is_first or conversation["title"] == "New chat":
        db.update_conversation(conversation_id, title=_title_from(message))

    # Project attachment: None = unchanged, "" = detach, otherwise attach.
    if request.project_id is not None:
        if request.project_id == "":
            db.update_conversation(conversation_id, project_id=None, project_name=None)
            conversation["project_id"] = None
        else:
            try:
                project_path(request.project_id)
            except (ValueError, FileNotFoundError) as error:
                yield {"type": "error", "message": f"Attached project is unavailable: {error}"}
                return
            db.update_conversation(
                conversation_id,
                project_id=request.project_id,
                project_name=request.project_name,
            )
            conversation["project_id"] = request.project_id

    project_id = conversation.get("project_id")
    mode = resolve_mode(request.mode, message, bool(project_id), conversation.get("mode"))
    db.update_conversation(conversation_id, mode=mode)

    # A pasted priced list + "calculate the total" is answered exactly, without the model.
    if mode == "chat" and not project_id:
        quick = try_total_list(message)
        if quick:
            label = "Parsed the list and added it up exactly"
            meta = {"mode": "chat", "model": "built-in calculator", "tools": [label]}
            yield {
                "type": "meta",
                "conversation_id": conversation_id,
                "title": db.get_conversation(conversation_id)["title"],
                "mode": "chat",
                "model": "built-in calculator",
                "project_id": None,
            }
            for i in range(0, len(quick["text"]), 40):
                yield {"type": "token", "text": quick["text"][i:i + 40]}
            message_id = db.add_message(conversation_id, "assistant", quick["text"], meta)
            yield {
                "type": "done",
                "response": quick["text"],
                "project_id": None,
                "stats": None,
                "message_id": message_id,
                "meta": meta,
            }
            return

    try:
        preferred = request.model or (config.CODE_MODEL if mode == "code" else config.CHAT_MODEL)
        alternates = [config.CODE_MODEL, config.CHAT_MODEL]
        model, warning = await resolve_model(preferred, alternates)
    except OllamaError as error:
        yield {"type": "error", "message": str(error)}
        return

    project_context = ""
    if project_id:
        try:
            project_context = await asyncio.to_thread(
                build_project_context, project_id, int(config.MAX_FILE_CHARS * 0.9)
            )
        except (ValueError, FileNotFoundError) as error:
            yield {"type": "warning", "text": f"The attached project could not be read: {error}"}
            project_id = None

    yield {
        "type": "meta",
        "conversation_id": conversation_id,
        "title": db.get_conversation(conversation_id)["title"],
        "mode": mode,
        "model": model,
        "project_id": project_id,
    }

    if warning:
        yield {"type": "warning", "text": warning}

    loaded = await loaded_models()
    if model not in loaded:
        yield {
            "type": "status",
            "text": f"Loading {model} into memory (first message after a break takes longer)…",
        }
    else:
        yield {"type": "status", "text": "Thinking…"}

    partial = ""
    tools_used = []
    saved = False

    try:
        async for event in run_agent_stream(
            message=message,
            history=history,
            project_id=project_id,
            mode=mode,
            model=model,
            project_context=project_context,
        ):
            if event["type"] == "token":
                partial += event["text"]
            elif event["type"] == "tool":
                tools_used.append(event["label"])
            elif event["type"] == "done":
                meta = {"mode": mode, "model": model, "tools": tools_used}
                if event.get("stats"):
                    meta["stats"] = event["stats"]
                message_id = db.add_message(
                    conversation_id, "assistant", event["response"], meta
                )
                saved = True
                event = {**event, "message_id": message_id, "meta": meta}
                if event.get("project_id") and event["project_id"] != project_id:
                    db.update_conversation(conversation_id, project_id=event["project_id"])
            yield event

    except OllamaError as error:
        yield {"type": "error", "message": str(error)}

    finally:
        # Stop button / closed tab: keep what was generated so far.
        if partial.strip() and not saved:
            db.add_message(
                conversation_id,
                "assistant",
                partial,
                {"mode": mode, "model": model, "tools": tools_used, "stopped": True},
            )


def _validate(request: ChatRequest):
    if not CONVERSATION_ID.match(request.conversation_id.strip()):
        raise HTTPException(status_code=400, detail="A valid conversation ID is required.")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    _validate(request)

    async def events():
        async for event in _run_chat(request):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming version of /chat/stream (same behaviour, one JSON answer)."""
    _validate(request)

    result = {"response": "", "conversation_id": request.conversation_id.strip()}

    async for event in _run_chat(request):
        if event["type"] == "meta":
            result["mode"] = event["mode"]
            result["model"] = event["model"]
        elif event["type"] == "done":
            result["response"] = event["response"]
        elif event["type"] == "error":
            result["response"] = event["message"]
            result["error"] = True

    return result
