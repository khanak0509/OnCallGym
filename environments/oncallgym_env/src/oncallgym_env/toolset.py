"""Tool server: lists the core tool schemas and forwards every call to the episode host.
No tool logic lives here (oncallgym.episode.Tools is the only implementation)."""

from typing import Any, cast

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent, Tool
from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.state import State

from oncallgym.episode import tool_schemas


class OnCallState(State):
    episode_url: str = ""
    episode_token: str = ""
    done: bool = False


class OnCallToolset(Toolset[SharedToolsetConfig, OnCallState]):
    TOOL_PREFIX = None

    def register(self, mcp: MCPServer) -> None:
        server = cast(Any, mcp)
        server.list_tools = self._with_state(self.list_tools)
        server.call_tool = self._with_state(self.call_tool)

    async def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name=s["function"]["name"],
                description=s["function"]["description"],
                input_schema=s["function"]["parameters"],
            )
            for s in tool_schemas()
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> CallToolResult:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.state.episode_url}/call",
                json={"name": name, "args": arguments or {}},
                headers={"X-Episode-Token": self.state.episode_token},
            )
        if r.status_code != 200:
            raise RuntimeError(f"episode server returned {r.status_code}")
        body = r.json()
        if body.get("done"):
            self.state.done = True
        return CallToolResult(
            content=[TextContent(type="text", text=body["output"])], is_error=False
        )


if __name__ == "__main__":
    OnCallToolset.run()
