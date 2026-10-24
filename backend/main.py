from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.chat import router as chat_router
from api.conversations import router as conversations_router
from api.projects import router as projects_router
from api.system import router as system_router
from services import db


app = FastAPI(title=config.APP_NAME)


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


db.init_db()

app.include_router(system_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(projects_router)
