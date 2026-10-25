import asyncio
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from services import db
from services.locks import ProjectBusy
from services.project_service import (
    project_path,
    add_uploaded_files,
    extract_project_with_stats,
    list_project_files,
    read_project_file,
    search_project,
    zip_files,
)


router = APIRouter()


def _safe_filename(name: str, default: str) -> str:
    cleaned = "".join(c for c in (name or "") if c.isalnum() or c in "-_. ").strip()
    return cleaned or default


@router.post("/upload-project")
async def upload_project(file: UploadFile = File(...)):
    try:
        stats = await asyncio.to_thread(extract_project_with_stats, file)

        return {
            "message": "Project uploaded successfully.",
            "project_id": stats["project_id"],
            "name": stats["name"],
            "files": stats["files"],
            "skipped": stats["skipped"],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save the project: {error}",
        )


@router.post("/upload-files")
async def upload_files(
    files: List[UploadFile] = File(...),
    project_id: str | None = Form(None),
):
    """Upload loose files. Adds to `project_id` if given, otherwise creates a new project."""
    try:
        stats = await asyncio.to_thread(
            add_uploaded_files,
            [(f.filename, f.file) for f in files],
            project_id or None,
        )
        return {**stats, "name": files[0].filename if len(files) == 1 else f"{len(files)} files"}

    except ProjectBusy as error:
        raise HTTPException(status_code=409, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save the files, so nothing was changed: {error}",
        )


@router.get("/projects/{project_id}/files")
async def get_project_files(project_id: str):
    try:
        files = list_project_files(project_id)

        return {
            "project_id": project_id,
            "files": files,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.get("/projects/{project_id}/file")
async def get_project_file(project_id: str, path: str):
    try:
        content = read_project_file(project_id, path)

        return {
            "project_id": project_id,
            "path": path,
            "content": content,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@router.get("/projects/{project_id}/search")
async def search_project_files(project_id: str, query: str):
    try:
        return search_project(project_id, query)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


class GeneratedFile(BaseModel):
    path: str
    content: str


class ZipRequest(BaseModel):
    files: List[GeneratedFile]
    name: str = "files"


@router.post("/download-zip")
async def download_zip(body: ZipRequest):
    """Bundle generated code blocks into a ZIP the browser can download."""
    try:
        data = await asyncio.to_thread(zip_files, [f.model_dump() for f in body.files])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    filename = _safe_filename(body.name, "files") + ".zip"

    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class TestsApproval(BaseModel):
    approved: bool


@router.post("/projects/{project_id}/tests-approval")
async def set_tests_approval(project_id: str, body: TestsApproval):
    """The user's explicit choice to let (or stop) Atlas running this project's tests.

    Running tests executes the project's own code, so it is off until this is set, and the
    model has no way to set it.
    """
    try:
        project_path(project_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    db.set_tests_approved(project_id, body.approved)
    return {"project_id": project_id, "tests_approved": body.approved}
