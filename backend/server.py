"""Canonical backend entrypoint.

This module provides a stable import path:
    uvicorn backend.server:app

It intentionally re-exports the existing FastAPI application from rag_server.py
to keep runtime behavior unchanged while repository structure is being cleaned up.
"""

from rag_server import app

__all__ = ["app"]

