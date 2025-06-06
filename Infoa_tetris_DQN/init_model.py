# init_model.py

import os
import torch
from networks.network import DQNConvNet  # 仮にネットワーク定義をここに置いた想定
from torch import nn

def init_and_save_model(
    save_path: str = "saved_models/dqn_tetris.pth",
    device: torch.device = torch.device("cpu"),
    in_channels: int = 1,
    height: int = 20,
    width: int = 10,
    n_actions: int = 6,
):
    """
    DQNConvNet を初期化し、save_path に state_dict を保存するサンプル。
    └─ save_path が存在しないディレクトリの場合は自動で作成します。
    """

    # 1) ディレクトリ作成チェック
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2) ネットワークの初期化
    #    DQNConvNet のイニシャライザは (in_channels, n_actions, height, width) を引数に受け取る想定です。
    #    height/width は内部で conv の形状計算に使う場合があるので渡すようにしてください。
    policy_net = DQNConvNet(
        in_channels=in_channels,
        n_actions=n_actions,
        height=height,
        width=width,
    ).to(device)

    # 3) 重み初期化（好みで Xavier や He などに変えてください）
    def weight_init(m: nn.Module):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    policy_net.apply(weight_init)

    # 4) state_dict を保存
    torch.save(policy_net.state_dict(), save_path)
    print(f"Model initialized and saved to {save_path}")

if __name__ == "__main__":
    # CPU 上で作成する場合の例
    init_and_save_model(
        save_path="saved_models/dqn_tetris.pth",
        device=torch.device("cpu"),
        in_channels=1,   # 盤面がシングルチャンネル (0:空 or 1-7:ブロックID)
        height=20,
        width=10,
        n_actions=6,     # 0:No-op,1:Left,2:Right,3:Rotate,4:Soft drop,5:Hard drop
    )