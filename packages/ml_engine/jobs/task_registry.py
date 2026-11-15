from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("jobs.task_registry")

TaskHandler = Callable[[dict[str, Any], Callable[[float], None]], dict[str, Any]]

_TASK_REGISTRY: dict[str, TaskHandler] = {}


def register_task(task_name: str):
    """Decorator to register a callable as an asynchronous task handler."""
    def decorator(fn: TaskHandler):
        _TASK_REGISTRY[task_name] = fn
        return fn
    return decorator


def get_task_handler(task_name: str) -> TaskHandler | None:
    return _TASK_REGISTRY.get(task_name)


def list_registered_tasks() -> list[str]:
    return list(_TASK_REGISTRY.keys())
