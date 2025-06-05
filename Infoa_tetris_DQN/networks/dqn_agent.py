# networks/dqn_agent.py

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .network import QNetwork
from .replay_buffer import ReplayBuffer, Transition 

class DQNAgent:
    """
    DQN エージェントのクラス。
    - policy_net: Q ネットワーク（現在のパラメータを持つ）
    - target_net: ターゲットネットワーク（一定間隔で policy_net の重みをコピー）
    - replay_buffer: 経験再生メモリ
    - optimizer: ネットワークを学習するための最適化器
    """

    def __init__(
        self,
        input_shape,           # (C, H, W)
        num_actions,           # action_space.n
        device,                # 'cpu' or 'cuda'
        lr=1e-4,               # 学習率
        gamma=0.99,            # 割引率
        buffer_capacity=100000,# リプレイバッファの容量
        batch_size=32,         # ミニバッチサイズ
        target_update_freq=1000,# ターゲットネットワーク更新間隔(ステップ数)
        epsilon_start=1.0,     # ε-greedy の初期 ε
        epsilon_end=0.01,      # ε の最小値
        epsilon_decay=1000000  # ε を減衰させるステップ数
    ):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # 1) ネットワーク定義
        self.policy_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net = QNetwork(input_shape, num_actions).to(self.device)
        # ターゲットネットワークは policy_net 初期状態をコピーしておく
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 推論専用に設定

        # 2) リプレイバッファの初期化
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # 3) オプティマイザ
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 4) ε-greedy 用変数
        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        """
        現在の state (NumPy あるいは Tensor) に対し、
        ε-greedy で行動を選択して返す。
        """
        # 1) ε の計算（線形もしくは指数関数的に decaying させる）
        eps_threshold = self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < eps_threshold:
            # ランダム行動
            action = random.randrange(self.num_actions)
            return action
        else:
            # ネットワークに状態を入力して最も大きい Q 値を持つ行動を返す
            # state: NumPy array なら Tensor に変換し、バッチ次元を追加
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            else:
                state_tensor = state.unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                q_values = self.policy_net(state_tensor)  # (1, num_actions)
                action = q_values.argmax(dim=1).item()
            return action

    def push_memory(self, state, action, reward, next_state, done):
        """
        リプレイバッファに (s, a, r, s', done) を保存するラッパー
        """
        # NumPy 配列のまま保存しておき、学習時に Tensor に変換しても良い
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_model(self):
        """
        バッファからサンプリングしてミニバッチ学習を行う。
        """
        if len(self.replay_buffer) < self.batch_size:
            # メモリ不足で学習できない場合はスキップ
            return

        # 1) ミニバッチ取得
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # batch.state: tuple of np.ndarray or Tensor, 長さ=batch_size
        # 以降、Tensor にまとめる処理

        # 2) Tensor への変換
        #    - states: (batch_size, C, H, W)
        #    - actions: (batch_size, 1)
        #    - rewards: (batch_size, 1)
        #    - next_states: (batch_size, C, H, W)
        #    - dones: (batch_size, 1) 0 or 1
        states = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 3) 現在の Q 値を policy_net から取得
        #    q_values: (batch_size, num_actions)
        #    現在バッチの行動に対応する Q 値だけ取り出す
        q_values = self.policy_net(states).gather(1, actions)  # (batch_size, 1)

        # 4) target_net を用いて次状態の max Q 値を取得
        #    next_q_values: (batch_size, 1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0].unsqueeze(1)
            # dones が True の場合、次状態の Q 値は 0 にする
            expected_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values

        # 5) ロス計算（均二乗誤差：MSE）
        loss = F.mse_loss(q_values, expected_q_values)

        # 6) optimizer でバックプロパゲーション
        self.optimizer.zero_grad()
        loss.backward()
        # Gradients clipping（必要に応じて）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 7) ターゲットネットワークの定期更新
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """
        モデルのパラメータを保存する（必要なら provide）。
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """
        保存済みモデルを読み込む。policy_net と target_net に読み込む。
        """
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())