import shutil

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import db
from services.project_service import WORKSPACE, PROJECT_ID_PATTERN


router = APIRouter()


class RenameRequest(BaseModel):
    title: str


@router.get("/conversations")
async def list_conversations():
    return {"conversations": db.list_conversations()}


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = db.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {**conversation, "messages": db.get_messages(conversation_id)}


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, body: RenameRequest):
    if not db.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    title = " ".join(body.title.split())[:80] or "New chat"
    db.update_conversation(conversation_id, title=title)
    return {"id": conversation_id, "title": title}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conversation = db.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    db.delete_conversation(conversation_id)

    # Remove the uploaded project too, unless another conversation still uses it.
    project_id = conversation.get("project_id")
    if (
        project_id
        and PROJECT_ID_PATTERN.match(project_id)
        and not db.project_in_use(project_id)
    ):
        shutil.rmtree(WORKSPACE / project_id, ignore_errors=True)

    return {"deleted": conversation_id}
