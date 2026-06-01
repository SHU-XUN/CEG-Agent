from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..llm import LLMClient, ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline.state import PipelineState
    from ..agent_loop.todo import TodoList


@dataclass
class ToolContext:
    
    state: "PipelineState"
    llm: LLMClient
    todos: "TodoList"
    finalize_request: Dict[str, Any] = field(default_factory=dict)
    sub_invoker: Optional[Callable[..., Dict[str, Any]]] = None
    bundle_dir: Optional[str] = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any], ToolContext], Dict[str, Any]]
    is_terminal: bool = False

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def run(self, arguments: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        return self.handler(arguments or {}, ctx)


NO_ARGS: Dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


def list_tool_names(tools: List[Tool]) -> List[str]:
    return [t.name for t in tools]
