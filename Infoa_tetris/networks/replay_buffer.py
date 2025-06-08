# networks/replay_buffer.py

import random
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    """
    DQN 用のリプレイバッファ（経験再生メモリ）。
    固定サイズのバッファに遷移を保存し、ランダムサンプリングする。
    """

    def __init__(self, capacity):
        """
        capacity: バッファに保持する最大遷移数
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        遷移をバッファに追加する
        state: NumPy 配列または Tensor
        action: int
        reward: float
        next_state: NumPy 配列または Tensor
        done: bool
        """
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        現在バッファに入っている遷移の数
        """
        return len(self.memory)