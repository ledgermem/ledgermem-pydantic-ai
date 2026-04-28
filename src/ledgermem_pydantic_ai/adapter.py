"""Wire LedgerMem search/add into a Pydantic AI ``Agent`` as tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from ledgermem import LedgerMem
from pydantic_ai import Agent, RunContext

DepsT = TypeVar("DepsT", bound="MemoryDeps")
OutT = TypeVar("OutT")


@dataclass
class MemoryDeps:
    """Per-run dependencies for the memory tools.

    Subclass this in your own ``Deps`` type if you want to carry additional
    state — only ``ledgermem`` and ``user_id`` are read by the adapter.
    """

    ledgermem: LedgerMem
    user_id: str
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def with_memory(
    agent: Agent[DepsT, OutT],
    *,
    search_limit: int = 5,
    search_name: str = "search_memory",
    add_name: str = "add_memory",
) -> Agent[DepsT, OutT]:
    """Register ``search_memory`` and ``add_memory`` tools on ``agent``.

    The agent's ``deps_type`` must be ``MemoryDeps`` or a subclass.
    Returns the same agent for chaining.
    """

    @agent.tool(name=search_name)
    async def _search(
        ctx: RunContext[DepsT],
        query: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search the user's long-term memory for relevant facts.

        Args:
            query: Natural-language description of what to recall.
            limit: Max results (defaults to the adapter's ``search_limit``).
        """
        deps = ctx.deps
        # Resolve the limit defensively — `limit or search_limit` treated 0
        # and negative numbers as "use default", which silently overrode
        # whatever the model asked for. Clamp into [1, 50].
        if limit is None or not isinstance(limit, int):
            requested = search_limit
        else:
            requested = max(1, min(50, limit))
        # Over-fetch so the post-retrieval user_id filter still leaves a
        # full page of results. Without filtering by user_id, search would
        # return memories that belong to OTHER users in the same workspace
        # — a privacy leak that prompt-injection can trivially exploit.
        fetch_limit = max(requested * 4, 20)
        raw = await _maybe_await(
            deps.ledgermem.search(query, limit=fetch_limit)
        )
        coerced = _coerce_results(raw)
        out: list[dict[str, Any]] = []
        # Reject ambiguous ownership outright. If the adapter was constructed
        # without a real user_id (None / "") we must NOT match memories whose
        # stored ``user_id`` is also None — that pattern silently returns
        # every legacy / unattributed memory to any caller missing deps.
        if not deps.user_id:
            return out
        for item in coerced:
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            stored_user_id = metadata.get("user_id")
            if stored_user_id is None or stored_user_id != deps.user_id:
                continue
            out.append(item)
            if len(out) >= requested:
                break
        return out

    @agent.tool(name=add_name)
    async def _add(
        ctx: RunContext[DepsT],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save a fact to the user's long-term memory.

        Args:
            content: The plain-text fact to remember.
            metadata: Optional structured tags merged into the stored entry.
        """
        deps = ctx.deps
        # Model-supplied metadata is merged FIRST so trusted server-controlled
        # fields (user_id, extra_metadata) cannot be overwritten by the LLM
        # via prompt injection.
        merged: dict[str, Any] = {
            **(metadata or {}),
            **deps.extra_metadata,
            "user_id": deps.user_id,
        }
        memory = await _maybe_await(
            deps.ledgermem.add(content, metadata=merged)
        )
        return _coerce_one(memory)

    return agent


# Re-exported so callers can attach the tools to a pre-built agent without
# importing the private decorators.
search_memory_tool = with_memory
add_memory_tool = with_memory


async def _maybe_await(value: Any) -> Any:
    """Support both sync and async LedgerMem clients."""
    if hasattr(value, "__await__"):
        return await value
    return value


def _coerce_results(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_coerce_one(item) for item in value]
    return [_coerce_one(value)]


def _coerce_one(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()  # type: ignore[no-any-return]
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    return {"value": item}
