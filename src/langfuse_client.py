"""Swap point for the real Langfuse cloud client.

Usage::

    from langfuse_client import get_client  # prefers real, falls back to shim

Set in .env (see .env.example)::

    LANGFUSE_HOST=https://cloud.langfuse.com
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...

If the real ``langfuse`` package is not installed (or keys are missing), this
returns the offline SQLite shim, so every demo still runs with zero config.
"""
from __future__ import annotations

import os


def get_client(db_path=None):
    """Return a Langfuse-compatible client: real SDK if available, else shim."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

    if public_key and secret_key:
        try:
            from langfuse import Langfuse  # type: ignore
        except ImportError:
            print("langfuse package not installed; using offline shim.")
        else:
            print(f"Using real Langfuse client against {host}.")
            return Langfuse(host=host, public_key=public_key,
                            secret_key=secret_key)

    from .shim import OfflineLangfuse
    print("Using offline SQLite shim (no keys needed).")
    return OfflineLangfuse(db_path)
