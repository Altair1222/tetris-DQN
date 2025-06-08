# networks/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    DQN 用のニューラルネットワーク。
    入力: 2 チャネルの盤面情報 (current_board + next_piece_mask)
         shape=(2, height, width), たとえば (2,20,10)
    出力: 各アクション価値（Q 値）のベクトル
    """

    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape  # チャネル数 (=2), 高さ (=20), 幅 (=10)

        # --- 畳み込み層を定義 ---
        # in_channels=c によって、自動的に 2 チャネル入力を受け付ける
        self.conv1 = nn.Conv2d(in_channels=c,   out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=3, stride=1, padding=1)

        # 畳み込み後の特徴マップのサイズを計算するために、
        # padding=1, kernel_size=3, stride=1 を 3 層通すと、
        # 高さも幅も変化しない（h × w のまま）が得られる想定です。
        # もしパディングを使わない場合は (h - 2*layers) のような計算が必要です。
        convh, convw = h, w

        # チャネル数は out_channels=64 なので、
        # 全結合層に渡す flat な次元サイズは (64 × convh × convw) になる
        linear_input_size = 64 * convh * convw

        # 全結合層
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        順伝播
        x: Tensor of shape (batch_size, 2, height, width)
        """
        x = F.relu(self.conv1(x))    # (batch, 32, h, w)
        x = F.relu(self.conv2(x))    # (batch, 64, h, w)
        x = F.relu(self.conv3(x))    # (batch, 64, h, w)

        # flatten
        x = x.view(x.size(0), -1)    # (batch, 64*h*w)
        x = F.relu(self.fc1(x))      # (batch, 512)
        q_values = self.fc2(x)       # (batch, num_actions)
        return q_values