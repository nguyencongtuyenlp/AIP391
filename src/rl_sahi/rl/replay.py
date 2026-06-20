from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self._frame = 0

        self._buffer: list = []
        self._priorities = np.zeros((capacity,), dtype=np.float64)
        self._pos = 0
        self._max_priority = 1.0

    @property
    def beta(self) -> float:
        frac = min(float(self._frame) / max(self.beta_frames, 1), 1.0)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def push(self, state, action, reward, next_state, done) -> None:
        data = (state, int(action), float(reward), next_state, bool(done))
        if len(self._buffer) < self.capacity:
            self._buffer.append(data)
        else:
            self._buffer[self._pos] = data
        self._priorities[self._pos] = self._max_priority ** self.alpha
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        self._frame += 1
        n = len(self._buffer)
        if n == 0 or batch_size <= 0:
            raise ValueError("Cannot sample from empty buffer")

        priorities = self._priorities[:n]
        probs = priorities / priorities.sum()

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        beta = self.beta
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self._buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td in zip(indices, td_errors):
            priority = (abs(float(td)) + 1e-6) ** self.alpha
            self._priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return len(self._buffer)
