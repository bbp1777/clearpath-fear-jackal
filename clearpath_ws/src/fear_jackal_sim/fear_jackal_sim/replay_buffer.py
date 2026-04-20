"""
Bounded replay-buffer helpers used by the trainer to retain recent transitions for logging,
relabeling, and offline reuse.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Iterable

from fear_jackal_sim.rl_types import Transition


class ExperienceReplayBuffer:
    """
    Simple deque-backed FIFO buffer for Transition objects.
    """
    def __init__(self, capacity: int) -> None:
        """
        Initialize the replay buffer with a fixed capacity.
        """
        self.capacity = int(capacity)
        self._buffer: deque[Transition] = deque(maxlen=self.capacity)

    def append(self, transition: Transition) -> None:
        """
        Append one transition to the buffer.
        """
        self._buffer.append(transition)

    def clear(self) -> None:
        """
        Remove every stored transition.
        """
        self._buffer.clear()

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Draw a random batch without replacement.
        """
        if batch_size <= 0:
            return []
        sample_size = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), sample_size)

    def extend(self, transitions: Iterable[Transition]) -> None:
        """
        Append a sequence of transitions.
        """
        for transition in transitions:
            self.append(transition)

    def as_list(self) -> list[Transition]:
        """
        Return the current buffer contents as a list.
        """
        return list(self._buffer)

    def __len__(self) -> int:
        """
        Return the number of stored transitions.
        """
        return len(self._buffer)
