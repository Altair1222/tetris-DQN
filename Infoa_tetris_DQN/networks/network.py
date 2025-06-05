# networks/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    DQN 用のニューラルネットワーク。
    入力: 環境の観測（例: Tetris の盤面マトリクス）
    出力: 各アクション価値（Q 値）のベクトル
    """

    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        # --- 例：簡易的な畳み込み＋全結合ネットワーク ---
        c, h, w = input_shape  # チャネル数, 高さ, 幅 (Tetris なら c=1, h=20, w=10)

        # 1) 畳み込み層
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # 畳み込み後の特徴量マップのサイズを計算（要調整）
        convw = w - 3 * 2  # 3層分の kernel_size=3, stride=1 の場合、(kernel_size-1)*3 = 2*3 = 6 だけ縮小
        convh = h - 3 * 2
        linear_input_size = convw * convh * 64  # チャネル 64 の特徴マップ

        # 2) 全結合層
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        順伝播
        x: Tensor of shape (batch_size, c, h, w)
        """
        # 1) 畳み込みと活性化
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 2) Flatten
        x = x.view(x.size(0), -1)

        # 3) 全結合
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)  # 出力次元は num_actions

        return q_values