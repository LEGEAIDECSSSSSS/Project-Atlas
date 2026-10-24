import os
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask

from services.project_service import (
    add_uploaded_files,
    apply_files,
    extract_project_with_stats,
    list_project_files,
    read_project_file,
    search_project,
    zip_files,
    zip_project,
)


router = APIRouter()


def _safe_filename(name: str, default: str) -> str:
    cleaned = "".join(c for c in (name or "") if c.isalnum() or c in "-_. ").strip()
    return cleaned or default


@router.post("/upload-project")
async def upload_project(file: UploadFile = File(...)):
    try:
        stats = extract_project_with_stats(file)

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


@router.post("/upload-files")
async def upload_files(
    files: List[UploadFile] = File(...),
    project_id: str | None = Form(None),
):
    """Upload loose files. Adds to `project_id` if given, otherwise creates a new project."""
    try:
        stats = add_uploaded_files(
            [(f.filename, f.file) for f in files],
            project_id=project_id or None,
        )
        return {**stats, "name": files[0].filename if len(files) == 1 else f"{len(files)} files"}

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


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


@router.get("/projects/{project_id}/download")
async def download_project(project_id: str, name: str = "project"):
    """Download the whole project (including any applied changes) as a ZIP."""
    try:
        path = zip_project(project_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    filename = _safe_filename(name, "project") + ".zip"

    return FileResponse(
        path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(os.unlink, path),
    )


class GeneratedFile(BaseModel):
    path: str
    content: str


class ApplyRequest(BaseModel):
    files: List[GeneratedFile]
    dry_run: bool = False


@router.post("/projects/{project_id}/apply")
async def apply_generated_files(project_id: str, body: ApplyRequest):
    """Write generated files into the project. `dry_run` only reports what would happen."""
    try:
        result = apply_files(
            project_id,
            [f.model_dump() for f in body.files],
            dry_run=body.dry_run,
        )
        return {"project_id": project_id, "dry_run": body.dry_run, "files": result}

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


class ZipRequest(BaseModel):
    files: List[GeneratedFile]
    name: str = "files"


@router.post("/download-zip")
async def download_zip(body: ZipRequest):
    """Bundle generated code blocks into a ZIP the browser can download."""
    try:
        data = zip_files([f.model_dump() for f in body.files])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    filename = _safe_filename(body.name, "files") + ".zip"

    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
