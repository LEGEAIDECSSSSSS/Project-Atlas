from fastapi import APIRouter

import config
from services.ollama import OllamaError, list_models, loaded_models

router = APIRouter()


@router.get("/health")
async def health():
    info = {
        "app_name": config.APP_NAME,
        "ollama": False,
        "models": [],
        "loaded": [],
        "chat_model": config.CHAT_MODEL,
        "code_model": config.CODE_MODEL,
        "num_ctx": config.NUM_CTX,
    }

    try:
        info["models"] = await list_models(force=True)
        info["loaded"] = await loaded_models()
        info["ollama"] = True
    except OllamaError as error:
        info["error"] = str(error)

    return info
