"""Installed command entry point.

The project still supports direct execution of modules from ``src/``. Keep
that legacy import mode behind this small adapter so importing the installed
entry point itself does not initialize Chroma, logging, or the MCP server.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    source_dir = Path(__file__).resolve().parent
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from rag_mcp_server import main as run_server

    run_server()
