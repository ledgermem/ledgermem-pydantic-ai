"""Smoke tests for the LedgerMem Pydantic AI adapter.

These tests exercise the public surface without making network calls — the
LedgerMem client is replaced with a fake.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ledgermem_pydantic_ai import MemoryDeps, with_memory


class FakeLedgerMem:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.add_calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        self.search_calls.append((query, limit))
        return [{"id": "m1", "content": f"hit for {query}"}]

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.add_calls.append((content, metadata or {}))
        return {"id": "m2", "content": content, "metadata": metadata or {}}


def test_memory_deps_dataclass_roundtrip() -> None:
    fake = FakeLedgerMem()
    deps = MemoryDeps(ledgermem=fake, user_id="u1", extra_metadata={"plan": "pro"})
    assert deps.user_id == "u1"
    assert deps.extra_metadata == {"plan": "pro"}
    assert deps.ledgermem is fake


def test_with_memory_registers_tools() -> None:
    """``with_memory`` registers two tools on the agent and returns it."""
    agent = MagicMock()
    registered: list[str] = []

    def tool_decorator(*, name: str):
        def wrap(fn):
            registered.append(name)
            return fn

        return wrap

    agent.tool = tool_decorator
    out = with_memory(agent)
    assert out is agent
    assert sorted(registered) == ["add_memory", "search_memory"]


def test_with_memory_custom_names() -> None:
    agent = MagicMock()
    registered: list[str] = []

    def tool_decorator(*, name: str):
        def wrap(fn):
            registered.append(name)
            return fn

        return wrap

    agent.tool = tool_decorator
    with_memory(agent, search_name="recall", add_name="remember")
    assert sorted(registered) == ["recall", "remember"]


@pytest.mark.asyncio
async def test_search_tool_calls_ledgermem() -> None:
    """The registered search tool delegates to ``LedgerMem.search``."""
    agent = MagicMock()
    captured: dict[str, Any] = {}

    def tool_decorator(*, name: str):
        def wrap(fn):
            captured[name] = fn
            return fn

        return wrap

    agent.tool = tool_decorator
    with_memory(agent, search_limit=3)

    fake = FakeLedgerMem()
    deps = MemoryDeps(ledgermem=fake, user_id="u1")
    ctx = MagicMock()
    ctx.deps = deps

    out = await captured["search_memory"](ctx, "coffee preference")
    assert out == [{"id": "m1", "content": "hit for coffee preference"}]
    assert fake.search_calls == [("coffee preference", 3)]


@pytest.mark.asyncio
async def test_add_tool_merges_metadata() -> None:
    agent = MagicMock()
    captured: dict[str, Any] = {}

    def tool_decorator(*, name: str):
        def wrap(fn):
            captured[name] = fn
            return fn

        return wrap

    agent.tool = tool_decorator
    with_memory(agent)

    fake = FakeLedgerMem()
    deps = MemoryDeps(
        ledgermem=fake, user_id="u1", extra_metadata={"plan": "pro"}
    )
    ctx = MagicMock()
    ctx.deps = deps

    out = await captured["add_memory"](
        ctx, "likes oat milk", metadata={"topic": "drinks"}
    )
    assert out["id"] == "m2"
    assert fake.add_calls == [
        ("likes oat milk", {"user_id": "u1", "plan": "pro", "topic": "drinks"})
    ]
