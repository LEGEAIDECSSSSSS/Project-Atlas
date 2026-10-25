import asyncio
import shutil

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import db
from services.locks import ProjectBusy, project_lock
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

    project_id = conversation.get("project_id")
    return {
        **conversation,
        "tests_approved": db.tests_approved(project_id) if project_id else False,
        "messages": db.get_messages(conversation_id),
    }


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, body: RenameRequest):
    if not db.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    title = " ".join(body.title.split())[:80] or "New chat"
    db.update_conversation(conversation_id, title=title)
    return {"id": conversation_id, "title": title}


def _remove_project(project_id):
    with project_lock(project_id, what="A test run or upload"):
        shutil.rmtree(WORKSPACE / project_id, ignore_errors=True)
        db.clear_project_flags(project_id)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conversation = db.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    project_id = conversation.get("project_id")
    removable = (
        project_id
        and PROJECT_ID_PATTERN.match(project_id)
        and not db.project_in_use(project_id, excluding=conversation_id)
    )

    # Never delete a project folder while tests or an upload are using it.
    if removable:
        try:
            await asyncio.to_thread(_remove_project, project_id)
        except ProjectBusy as error:
            raise HTTPException(status_code=409, detail=str(error))

    db.delete_conversation(conversation_id)
    return {"deleted": conversation_id}
