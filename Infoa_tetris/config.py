# config.py

import torch

# --- 環境設定 ---
ENV_NAME = "TetrisEnv-v0"  # 実際には gymnasium.register で登録した ID

# --- 学習ハイパーパラメータ ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 30000  # 1エピソードあたりの最大ステップ数
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 1000  # ステップ数ベースでターゲット更新

# --- 報酬設定 ---
REWARD_ALIVE = 0
REWARD_1LINE = 40
REWARD_2LINE = 100
REWARD_3LINE = 300
REWARD_4LINE = 1200
REWARD_HARD_DROP = 2
REWARD_GAME_OVER = -20

# --- バッファ設定 ---
HEAP_CAPACITY = 100

# ε-greedy のパラメータ
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 1000000

#ビジュアライズの頻度
VISUALIZE_EVERY = 100  # 100エピソードごとに可視化
# モデル保存ディレクトリ
CHECKPOINT_DIR = "checkpoints"

# ログ出力先
LOG_DIR = "logs"