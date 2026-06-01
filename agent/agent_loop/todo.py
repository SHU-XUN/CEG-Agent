from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


_VALID_STATUS = {"pending", "in_progress", "completed"}


@dataclass
class TodoItem:
    id: str
    content: str
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "content": self.content, "status": self.status}


@dataclass
class TodoList:
    items: List[TodoItem] = field(default_factory=list)

    def upsert(self, raw: Dict[str, Any]) -> None:
        item_id = str(raw.get("id"))
        content = str(raw.get("content", ""))
        status = str(raw.get("status", "pending"))
        if status not in _VALID_STATUS:
            status = "pending"
        for it in self.items:
            if it.id == item_id:
                it.content = content
                it.status = status
                return
        self.items.append(TodoItem(id=item_id, content=content, status=status))

    def replace(self, raws: List[Dict[str, Any]]) -> None:
        self.items = []
        for raw in raws:
            self.upsert(raw)

    def snapshot(self) -> List[Dict[str, Any]]:
        return [it.to_dict() for it in self.items]

    def all_done(self) -> bool:
        return bool(self.items) and all(
            it.status == "completed" for it in self.items
        )
