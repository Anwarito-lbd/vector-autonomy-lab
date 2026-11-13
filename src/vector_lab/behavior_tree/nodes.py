"""Composable Behavior-Tree primitives: Sequence, Selector, Parallel, Inverter & leaves."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum

from vector_lab.types import Blackboard


class NodeStatus(StrEnum):
    """Standard Behavior-Tree tick return status."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class BTNode(ABC):
    """Abstract base node in a composable Behavior Tree."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def tick(self, blackboard: Blackboard) -> NodeStatus:
        """Evaluate this node against the shared blackboard."""

    def reset(self) -> None:  # noqa: B027
        """Reset any stateful execution cursors or child node states."""


class Condition(BTNode):
    """Stateless leaf evaluating a boolean predicate on the blackboard."""

    def __init__(self, name: str, predicate: Callable[[Blackboard], bool]) -> None:
        super().__init__(name)
        self.predicate = predicate

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        return NodeStatus.SUCCESS if self.predicate(blackboard) else NodeStatus.FAILURE


class Action(BTNode):
    """Leaf node executing a robot/blackboard action returning NodeStatus."""

    def __init__(self, name: str, fn: Callable[[Blackboard], NodeStatus]) -> None:
        super().__init__(name)
        self.fn = fn

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        return self.fn(blackboard)


class Inverter(BTNode):
    """Decorator inverting SUCCESS <-> FAILURE while passing RUNNING unchanged."""

    def __init__(self, name: str, child: BTNode) -> None:
        super().__init__(name)
        self.child = child

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        status = self.child.tick(blackboard)
        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        if status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING

    def reset(self) -> None:
        self.child.reset()


class Sequence(BTNode):
    """Composite AND node: ticks children sequentially; reactive by default (memory=False)."""

    def __init__(self, name: str, children: list[BTNode], memory: bool = False) -> None:
        super().__init__(name)
        self.children = children
        self.memory = memory
        self._running_index = 0

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        start_idx = self._running_index if self.memory else 0
        for idx in range(start_idx, len(self.children)):
            status = self.children[idx].tick(blackboard)
            if status == NodeStatus.RUNNING:
                self._running_index = idx if self.memory else 0
                return NodeStatus.RUNNING
            if status == NodeStatus.FAILURE:
                self._running_index = 0
                return NodeStatus.FAILURE
        self._running_index = 0
        return NodeStatus.SUCCESS

    def reset(self) -> None:
        self._running_index = 0
        for child in self.children:
            child.reset()


class Selector(BTNode):
    """Composite priority fallback node: returns on first non-FAILURE child."""

    def __init__(self, name: str, children: list[BTNode], memory: bool = False) -> None:
        super().__init__(name)
        self.children = children
        self.memory = memory
        self._running_index = 0

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        start_idx = self._running_index if self.memory else 0
        for idx in range(start_idx, len(self.children)):
            status = self.children[idx].tick(blackboard)
            if status == NodeStatus.RUNNING:
                self._running_index = idx if self.memory else 0
                return NodeStatus.RUNNING
            if status == NodeStatus.SUCCESS:
                self._running_index = 0
                return NodeStatus.SUCCESS
        self._running_index = 0
        return NodeStatus.FAILURE

    def reset(self) -> None:
        self._running_index = 0
        for child in self.children:
            child.reset()


class Parallel(BTNode):
    """Composite concurrent node ticking all children and aggregating by policy thresholds."""

    def __init__(
        self,
        name: str,
        children: list[BTNode],
        success_threshold: int | None = None,
        failure_threshold: int = 1,
    ) -> None:
        super().__init__(name)
        self.children = children
        self.success_threshold = (
            len(children) if success_threshold is None else success_threshold
        )
        self.failure_threshold = failure_threshold

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        successes = 0
        failures = 0
        for child in self.children:
            status = child.tick(blackboard)
            if status == NodeStatus.SUCCESS:
                successes += 1
            elif status == NodeStatus.FAILURE:
                failures += 1

        if failures >= self.failure_threshold:
            return NodeStatus.FAILURE
        if successes >= self.success_threshold:
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING

    def reset(self) -> None:
        for child in self.children:
            child.reset()
