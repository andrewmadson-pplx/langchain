"""LangChain integration for the Model Context Protocol (MCP).

Convert the tools exposed by an MCP server into native LangChain tools. Connections are
managed by [FastMCP](https://gofastmcp.com); this package only converts tools. Requires
the optional `mcp` dependency:

```bash
pip install "langchain[mcp]"
```

Example:
    ```python
    from fastmcp import Client

    from langchain.mcp import load_mcp_tools

    async with Client("https://example.com/mcp") as client:
        tools = await load_mcp_tools(client)
    ```
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("fastmcp") is None:  # pragma: no cover
    msg = (
        "langchain.mcp requires the optional 'mcp' dependency. "
        'Install it with: pip install "langchain[mcp]"'
    )
    raise ModuleNotFoundError(msg)

from langchain.mcp.tools import convert_mcp_tool, load_mcp_tools

__all__ = [
    "convert_mcp_tool",
    "load_mcp_tools",
]
