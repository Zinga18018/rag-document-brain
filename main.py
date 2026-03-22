"""
RAG Document Brain
==================
Upload documents, chunk and embed them with Sentence-Transformers,
store in ChromaDB, then ask questions and get grounded answers
with source citations and similarity scores.

Usage:
    python main.py
    # Then open http://localhost:8002/docs
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import RAGPipeline, RAGConfig
from api import register_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

config = RAGConfig()
pipeline = RAGPipeline(config)

app = FastAPI(
    title="RAG Document Brain",
    description="Retrieval-Augmented Generation with Sentence-Transformers + ChromaDB",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app, pipeline)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.port, reload=True)
