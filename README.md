# ledgermem-pydantic-ai

LedgerMem adapter for [Pydantic AI](https://ai.pydantic.dev). One call wires
`search_memory` and `add_memory` tools onto any `Agent`, with per-user
metadata pulled from `RunContext.deps`.

## Install

```bash
pip install ledgermem-pydantic-ai
```

## Quickstart (30 seconds)

```python
from ledgermem import LedgerMem
from pydantic_ai import Agent
from ledgermem_pydantic_ai import MemoryDeps, with_memory

agent = Agent(
    "openai:gpt-4o",
    deps_type=MemoryDeps,
    system_prompt=(
        "Use search_memory before answering, and add_memory when the user "
        "shares something worth remembering."
    ),
)
with_memory(agent, search_limit=5)

deps = MemoryDeps(
    ledgermem=LedgerMem(api_key="...", workspace_id="..."),
    user_id="user-42",
)

result = await agent.run("Do you remember what coffee I like?", deps=deps)
print(result.data)
```

## What it adds

| Tool            | Description                                      |
| --------------- | ------------------------------------------------ |
| `search_memory` | Top-k semantic recall scoped by `deps.user_id`   |
| `add_memory`    | Persists a fact, merging `deps.extra_metadata`   |

Both tool names are configurable via `with_memory(agent, search_name=..., add_name=...)`.

## License

MIT
