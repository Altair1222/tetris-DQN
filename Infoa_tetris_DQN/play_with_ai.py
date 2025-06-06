# play_with_ai.py

import time
import torch
import numpy as np
from networks.network import DQNConvNet
from env.tetris_env import TetrisEnv  # 先ほど作成した環境
from collections import deque

def load_model(
    model: DQNConvNet,
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
):
    """
    checkpoint_path から重みをロードして model にセットするユーティリティ。
    """
    if not torch.cuda.is_available():
        device = torch.device("cpu")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 推論モードに切り替え
    return model

def select_action_greedy(state: np.ndarray, policy_net: DQNConvNet, device: torch.device):
    """
    ε-グリーディーではなく、常に Q 値が最大の行動を返す関数。
    state: (height, width) の NumPy array を受け取り、Tensor に変換して model に通す。
    """
    # 1) NumPy -> Float Tensor (batch=1, channel=1, H, W)
    state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float().to(device)
    # 2) 推論
    with torch.no_grad():
        q_values = policy_net(state_tensor)  # (1, n_actions)
    # 3) 最大 Q 値のインデックスを取得
    action = q_values.argmax(dim=1).item()
    return action

def play_episode(env: TetrisEnv, policy_net: DQNConvNet, device: torch.device):
    """
    1 エピソード分プレイを行い、最終スコアとライン数を返す。
    env.render() を使うことで可視化も可能。
    """
    state, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # 1) Greedy 行動選択
        action = select_action_greedy(state, policy_net, device)

        # 2) 環境を 1 ステップ進める
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # 3) 可視化 (human モード)
        env.render(mode="human")
        # ループが速すぎる場合は少しだけ待機
        time.sleep(1 / env.metadata["render_fps"])

        state = next_state

    # 最後にウィンドウを閉じる（必要なら）
    plt = env._fig.canvas.manager.window.__class__.__module__.split('.')[0]
    try:
        env._fig.canvas.manager.window.close()
    except Exception:
        pass

    return info.get("score", 0), info.get("lines_cleared", 0)

def main(
    model_path: str = "saved_models/dqn_tetris.pth",
    device_str: str = "cpu",
    num_episodes: int = 5,
):
    # -- デバイスの設定 --
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- 環境の初期化 --
    env = TetrisEnv(
        height=20,
        width=10,
        render_mode="human",
        fig_width=8,     # 任意でウィンドウサイズを変更可能
        fig_height=5,
    )

    # -- モデルのロード/初期化 --
    # DQNConvNet の引数は、イニシャライズ時と合わせる必要があります
    policy_net = DQNConvNet(
        in_channels=1,
        n_actions=env.action_space.n,
        height=env.height,
        width=env.width,
    )
    policy_net = load_model(policy_net, model_path, device)

    # -- エピソードループ --
    for ep in range(num_episodes):
        score, lines = play_episode(env, policy_net, device)
        print(f"Episode {ep + 1} → Score: {score}, Lines Cleared: {lines}")

    # 最後に環境を閉じて終了
    env.close()

if __name__ == "__main__":
    # GPU を使いたい場合は "cuda"、CPU のみなら "cpu" にします
    main(model_path="saved_models/dqn_tetris.pth", device_str="cpu", num_episodes=5)