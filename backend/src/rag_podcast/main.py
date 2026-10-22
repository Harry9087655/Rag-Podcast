from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.audio import router as audio_router
from .api.health import router as health_router
from .api.query import router as query_router
from .ingestion.router import router as ingestion_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(ingestion_router)
app.include_router(health_router)
app.include_router(query_router)
app.include_router(audio_router)