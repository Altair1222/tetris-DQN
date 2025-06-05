# train.py
import matplotlib.pyplot as plt
import os
import time
import random
import numpy as np
import torch
import gymnasium as gym

from env.tetris_env import TetrisEnv
from networks.dqn_agent import DQNAgent
from utils.utils import save_model_checkpoint
import config

def main():
    """"
    # -------------------------
    # 1) 乱数シードの固定
    # -------------------------
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    """
    # -------------------------
    # 2) 環境の登録と生成
    # -------------------------
    # カスタム環境として TetrisEnv を登録しておく例（必要なら）
    # gymnasium.register(
    #     id=config.ENV_NAME,
    #     entry_point="envs.tetris_env:TetrisEnv",
    # )
    # env = gym.make(config.ENV_NAME)

    # ここでは直接クラスを呼び出しても良い
    env = TetrisEnv(render_mode="human")
    # 観測空間の形状を取得 (例: (20,10) → (1,20,10) のようにチャネルを付与)
    obs_shape = env.observation_space.shape  # (20,10)
    # PyTorch NN の入力として (C, H, W) に合わせる
    # Tetris データを [C, H, W] にするため、C=1 を追加
    input_shape = (1, obs_shape[0], obs_shape[1])
    num_actions = env.action_space.n

    # -------------------------
    # 3) エージェントの初期化
    # -------------------------
    agent = DQNAgent(
        input_shape=input_shape,
        num_actions=num_actions,
        device=config.DEVICE,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        buffer_capacity=config.BUFFER_CAPACITY,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ,
        epsilon_start=config.EPS_START,
        epsilon_end=config.EPS_END,
        epsilon_decay=config.EPS_DECAY
    )
    visualize_every = 10  # 50 エピソードに 1 回、render を呼ぶ
    # モデル保存用ディレクトリ作成
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # -------------------------
    # 4) メイン学習ループ
    # -------------------------
    for episode in range(1, config.NUM_EPISODES + 1):
        # 4-1) 環境のリセット
        state, info = env.reset()
        # state: (20,10) の NumPy array → Tensor に変換するときにチャネル次元を付与
        state = np.expand_dims(state, axis=0)  # → (1,20,10)

        total_reward = 0.0
 
      
        for t in range(config.MAX_STEPS_PER_EPISODE):
            if (episode + 1) % visualize_every == 0:
                env.render()
            # 4-2) ε-greedy で行動選択
            action = agent.select_action(state)

            # 4-3) 環境を 1 ステップ進める
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)  # (1,20,10)
            total_reward += reward

            # 4-4) 経験をリプレイバッファに保存
            agent.push_memory(state, action, reward, next_state, float(terminated))

            # 4-5) モデルの更新（ミニバッチ学習）
            agent.update_model()

            # 4-6) 次状態を現在状態にする
            state = next_state

     
            # 4-7) エピソード終了判定
            if terminated or truncated:
                break

        # -------------------------
        # 5) エピソードごとのログ出力
        # -------------------------
        print(f"Episode {episode} \tTotal Reward: {total_reward:.2f} \tSteps: {t+1}")

        # -------------------------
        # 6) 一定エピソードごとにモデルを保存
        # -------------------------
        if episode % 100 == 0:
            save_model_checkpoint(agent, config.CHECKPOINT_DIR, episode)
        


            

    # -------------------------
    # 7) 学習終了後の処理
    # -------------------------
    # 最終モデルを保存
    save_model_checkpoint(agent, config.CHECKPOINT_DIR, "final")
        # 50 エピソードに 1 回だけ可視化



    env.close()

if __name__ == "__main__":
    main()