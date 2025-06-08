# train.py
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch
import gymnasium as gym
import heapq

from env.tetris_env import TetrisEnv
from networks.dqn_agent import DQNAgent
from utils.utils import save_model_checkpoint
import config

def main():

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
    input_shape = (2, obs_shape[0], obs_shape[1])
    num_actions = env.action_space.n
    heap = []  # empty list
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
    # モデル保存用ディレクトリ作成
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
   
    # -------------------------
    # 4) メイン学習ループ
    # -------------------------
    for episode in range(1, config.NUM_EPISODES + 1):
        # 4-1) 環境のリセット
        state, info = env.reset()
        # state is already shape (2,20,10)

        total_reward = 0.0
      
        for t in range(config.MAX_STEPS_PER_EPISODE):
                # 4-2) ε-greedy で行動選択
            action = agent.select_action(state, training = True)
            # 4-3) 環境を 1 ステップ進める
            next_state, reward, terminated, truncated, info = env.step(action)
            # next_state is already shape (2,20,10)
            total_reward += reward  # 生存報酬を加算

            # 4-4) 経験をリプレイバッファに保存
            """
            if len(heap) < config.HEAP_CAPACITY:
            # ヒープにまだ空きがある場合はそのまま push
                heapq.heappush(heap,reward)
                agent.push_memory(state, action, reward, next_state, float(terminated))
            else:
            # capacity 個すでにある → いまある最小報酬と比較
            # self._heap[0][0] が現在ヒープ内で最も小さい報酬
                if reward > heap[0]:
                # 新しい報酬 reward がヒープ中の最小より大きいなら、
                # pop してから新しいタプルを push する
                    heapq.heapreplace(heap,reward)
                    agent.push_memory(state, action, reward, next_state, float(terminated))
            # そうでない場合は、上位 100 に入らないので破棄
"""
            agent.push_memory(state, action, reward, next_state, float(terminated))
            # 4-5) モデルの更新（ミニバッチ学習）
            agent.update_model()

            # 4-6) 次状態を現在状態にする
            state = next_state
     
            # 4-7) エピソード終了判定
            if terminated or truncated:
                break

        if( episode % config.VISUALIZE_EVERY == 0):
            state, info = env.reset()
            # state is already shape (2,20,10)
            for t in range(config.MAX_STEPS_PER_EPISODE):
                env.render()
                action = agent.select_action(state, training = False)
                # 4-3) 環境を 1 ステップ進める
                next_state, reward, terminated, truncated, info = env.step(action)
                # next_state is already shape (2,20,10)

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

    env.close()

if __name__ == "__main__":
    main()