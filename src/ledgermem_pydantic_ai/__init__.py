"""Mnemo adapter for Pydantic AI agents."""

from .adapter import (
    MemoryDeps,
    add_memory_tool,
    search_memory_tool,
    with_memory,
)

__all__ = [
    "MemoryDeps",
    "add_memory_tool",
    "search_memory_tool",
    "with_memory",
]
__version__ = "0.1.0"
