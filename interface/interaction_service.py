"""Interaction service coordinating human-in-the-loop workflows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class InteractionTask:
    task_id: str
    task_type: str
    description: str
    payload: Dict
    status: str = "pending"
    history: List[str] = field(default_factory=list)


class InteractionService:
    def __init__(self) -> None:
        self._queue: Deque[InteractionTask] = deque()
        self._counter = 1

    def enqueue(self, task_type: str, description: str, payload: Dict) -> InteractionTask:
        task = InteractionTask(
            task_id=f"task-{self._counter}",
            task_type=task_type,
            description=description,
            payload=payload,
        )
        self._counter += 1
        self._queue.append(task)
        return task

    def list_tasks(self) -> List[InteractionTask]:
        return list(self._queue)

    def resolve(self, task_id: str, note: str) -> None:
        for task in self._queue:
            if task.task_id == task_id:
                task.status = "resolved"
                task.history.append(note)
                break

    def next_pending(self) -> Optional[InteractionTask]:
        for task in self._queue:
            if task.status == "pending":
                return task
        return None
