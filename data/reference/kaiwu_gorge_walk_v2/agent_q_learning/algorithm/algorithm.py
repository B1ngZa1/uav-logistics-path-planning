#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import random
import numpy as np
from collections import deque


_USE_CONFIGURED_TAU = object()


class ReplayBuffer:
    """
    Circular replay buffer for experience replay.

    Stores (state, action, reward, next_state, done) transitions and supports
    uniform random sampling for mini-batch Q-learning updates.
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done=0):
        self.buffer.append((
            int(state), int(action), float(reward), int(next_state), float(done)
        ))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        states = np.array([x[0] for x in batch], dtype=np.int64)
        actions = np.array([x[1] for x in batch], dtype=np.int32)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch], dtype=np.int64)
        dones = np.array([x[4] for x in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class SparseQTable:
    """
    Sparse Q table for discrete position and treasure-mask features.

    Expanded feature engineering makes the theoretical state space large. A
    dense NumPy array would waste memory because training only visits a small
    subset of states. This table lazily creates Q rows when a state is used.
    """

    def __init__(self, action_size, default_value=1.0):
        self.action_size = action_size
        self.default_value = default_value
        self.data = {}

    def row(self, state):
        state = int(state)
        if state not in self.data:
            self.data[state] = np.ones(self.action_size, dtype=np.float32) * self.default_value
        return self.data[state]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            state, action = key
            row = self.row(state)
            if isinstance(action, slice):
                return row[action]
            return row[int(action)]
        return self.row(key)

    def __setitem__(self, key, value):
        state, action = key
        row = self.row(state)
        if isinstance(action, slice):
            row[action] = value
        else:
            row[int(action)] = value

    # ------------------------------------------------------------------
    # Batch operations for vectorized Double Q-Learning updates
    # ------------------------------------------------------------------

    def batch_get(self, states, actions):
        """Return Q[state, action] for each (state, action) pair as a 1-D array."""
        values = np.empty(len(states), dtype=np.float32)
        for i in range(len(states)):
            s = int(states[i])
            a = int(actions[i])
            if s in self.data:
                values[i] = self.data[s][a]
            else:
                values[i] = self.default_value
        return values

    def batch_get_row(self, states):
        """Return (N, action_size) array of all Q-values for each state."""
        rows = np.empty((len(states), self.action_size), dtype=np.float32)
        for i, s in enumerate(states):
            s = int(s)
            if s in self.data:
                rows[i] = self.data[s]
            else:
                rows[i] = self.default_value
        return rows

    def batch_update_mean(self, states, actions, deltas):
        """Apply the mean delta once for every unique state-action pair.

        All TD errors in a sampled batch are calculated from the same Q-table
        snapshot. Applying every duplicate independently would therefore make
        the effective learning rate depend on its accidental batch frequency.
        """
        grouped = {}
        for state, action, delta in zip(states, actions, deltas):
            key = (int(state), int(action))
            total, count = grouped.get(key, (0.0, 0))
            grouped[key] = (total + float(delta), count + 1)

        for (s, a), (total, count) in grouped.items():
            if s not in self.data:
                self.data[s] = np.ones(self.action_size, dtype=np.float32) * self.default_value
            self.data[s][a] += total / count

    # ------------------------------------------------------------------

    def to_serializable(self):
        return {
            "action_size": self.action_size,
            "default_value": self.default_value,
            "data": self.data,
        }

    @classmethod
    def from_serializable(cls, payload, action_size, default_value=1.0):
        table = cls(action_size=action_size, default_value=default_value)
        if isinstance(payload, dict) and "data" in payload:
            saved_action_size = int(payload.get("action_size", action_size))
            if saved_action_size != int(action_size):
                raise ValueError(
                    f"checkpoint action size {saved_action_size} does not match {action_size}"
                )
            table.action_size = saved_action_size
            table.default_value = float(payload.get("default_value", default_value))
            for key, value in payload["data"].items():
                row = np.asarray(value, dtype=np.float32)
                if row.shape != (saved_action_size,):
                    raise ValueError(f"invalid Q row shape for state {key}: {row.shape}")
                table.data[int(key)] = row
            return table
        if isinstance(payload, np.ndarray) and payload.ndim == 2:
            if payload.shape[1] != int(action_size):
                raise ValueError(f"invalid dense Q shape: {payload.shape}")
            for state in range(payload.shape[0]):
                row = payload[state]
                if not np.all(row == default_value):
                    table.data[state] = np.asarray(row, dtype=np.float32)
            return table
        raise ValueError("checkpoint does not contain a compatible Q table")


class Algorithm:
    def __init__(self, gamma, learning_rate, state_size, action_size,
                 replay_capacity=100000, batch_size=64, target_update_freq=500,
                 lr_min=0.001, lr_decay_steps=50000, tau=None):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.lr_min = lr_min
        self.lr_decay_steps = lr_decay_steps
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Soft-update coefficient (None = hard sync)
        self.tau = tau

        # Double Q-Learning: Q for action selection, Q_target for evaluation
        self.Q = SparseQTable(action_size=self.action_size)
        self.Q_target = SparseQTable(action_size=self.action_size)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # Step counter for LR decay and target network sync
        self.train_steps = 0

        # Moving average of TD error magnitude for stability tracking
        self.loss_ema = 0.0

    def _current_lr(self):
        """Exponential decay from initial_lr to lr_min over lr_decay_steps."""
        if self.train_steps >= self.lr_decay_steps:
            return self.lr_min
        decay_ratio = self.train_steps / self.lr_decay_steps
        return self.initial_lr * (self.lr_min / self.initial_lr) ** decay_ratio

    def _double_q_update(self, states, actions, rewards, next_states, dones, lr):
        """
        Vectorized Double Q-Learning batch update.

        Q(s,a) += lr * (r + gamma * Q_target(s', argmax_a' Q(s', a')) * (1-done) - Q(s,a))

        Uses Q for action selection and Q_target for value evaluation to
        reduce overestimation bias. All TD targets are computed in a single
        batch operation.
        """
        # Batch lookup: Q(s, a) for every sample
        q_current = self.Q.batch_get(states, actions)

        # Batch lookup: full row Q(s') for every next state
        q_next_rows = self.Q.batch_get_row(next_states)                     # (N, action_size)

        # Best actions according to Q (action selection)
        best_next_actions = np.argmax(q_next_rows, axis=1).astype(np.int32)  # (N,)

        # Evaluate best actions using Q_target (value evaluation)
        q_target_next = self.Q_target.batch_get(next_states, best_next_actions)  # (N,)

        # TD targets with done flag: no future value for terminal states
        td_targets = rewards + self.gamma * q_target_next * (1.0 - dones)

        # TD errors
        td_errors = td_targets - q_current

        # Apply one mean update per unique pair. Repeated replay samples must
        # not silently multiply the configured learning rate.
        self.Q.batch_update_mean(states, actions, lr * td_errors)

        return td_errors

    def learn(self, list_sample_data):
        """
        Q-Learning with experience replay and Double Q-Learning.

        Stores samples in replay buffer, then performs mini-batch updates.
        Periodically syncs target network Q_target ← Q (hard or soft).

        Args:
            list_sample_data: List of sample dicts with state, action, reward,
                              next_state, and optionally done.
        """
        # Push all incoming samples to replay buffer
        for sample in list_sample_data:
            self.replay_buffer.push(
                sample["state"],
                sample["action"],
                sample["reward"],
                sample["next_state"],
                sample.get("done", 0),
            )

        # Wait until we have enough samples for a batch
        if len(self.replay_buffer) < self.batch_size:
            return {
                "loss": float(self.loss_ema),
                "td_error": float(self.loss_ema),
                "lr": self._current_lr(),
            }

        # Sample mini-batch and perform Double Q-Learning update
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        current_lr = self._current_lr()
        td_errors = self._double_q_update(states, actions, rewards, next_states, dones, current_lr)
        self.train_steps += 1

        # Mean absolute TD error measures value-estimation drift. Huber loss
        # remains readable when sparse treasure rewards produce large errors.
        mean_td_error = float(np.mean(np.abs(td_errors)))
        abs_td_errors = np.abs(td_errors)
        huber_losses = np.where(
            abs_td_errors <= 1.0,
            0.5 * np.square(td_errors),
            abs_td_errors - 0.5,
        )
        mean_loss = float(np.mean(huber_losses))

        # Update moving average of loss
        self.loss_ema = 0.95 * self.loss_ema + 0.05 * mean_td_error

        # Periodic target network sync
        if self.train_steps % self.target_update_freq == 0:
            self.sync_target()

        return {"loss": mean_loss, "td_error": mean_td_error, "lr": current_lr}

    def sync_target(self, tau=_USE_CONFIGURED_TAU):
        """
        Sync target network.

        If tau is None (or not provided), performs a hard copy: Q_target = Q.
        If tau is a float in (0, 1], performs Polyak soft update:
            Q_target = tau * Q + (1 - tau) * Q_target

        Args:
            tau: Optional soft-update coefficient. Falls back to self.tau.
        """
        if tau is _USE_CONFIGURED_TAU:
            tau = self.tau

        if tau is None:
            # Hard sync: copy all Q data to Q_target
            self.Q_target.action_size = self.Q.action_size
            self.Q_target.default_value = self.Q.default_value
            self.Q_target.data = {
                state: row.copy()
                for state, row in self.Q.data.items()
            }
        else:
            # Soft Polyak averaging: Q_target <- tau * Q + (1-tau) * Q_target
            tau_f = float(tau)
            inv_tau = 1.0 - tau_f

            if not 0.0 < tau_f <= 1.0:
                raise ValueError(f"tau must be in (0, 1], got {tau_f}")

            q_default = np.full(self.action_size, self.Q.default_value, dtype=np.float32)
            target_default = np.full(
                self.action_size, self.Q_target.default_value, dtype=np.float32
            )
            blended_default_value = (
                inv_tau * self.Q_target.default_value + tau_f * self.Q.default_value
            )
            blended_default = np.full(
                self.action_size, blended_default_value, dtype=np.float32
            )
            updated = {}
            for state in set(self.Q.data) | set(self.Q_target.data):
                q_row = self.Q.data.get(state, q_default)
                target_row = self.Q_target.data.get(state, target_default)
                blended = inv_tau * target_row + tau_f * q_row
                if not np.allclose(blended, blended_default):
                    updated[state] = np.asarray(blended, dtype=np.float32)
            self.Q_target.default_value = float(blended_default_value)
            self.Q_target.data = updated
