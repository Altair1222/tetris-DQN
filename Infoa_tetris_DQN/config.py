# config.py

import torch

# --- 環境設定 ---
ENV_NAME = "TetrisEnv-v0"  # 実際には gymnasium.register で登録した ID

# --- 学習ハイパーパラメータ ---
SEED = 123
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 20000
MAX_STEPS_PER_EPISODE = 10000  # 1エピソードあたりの最大ステップ数
LEARNING_RATE = 0.99
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_CAPACITY = 100000
TARGET_UPDATE_FREQ = 1000  # ステップ数ベースでターゲット更新

# ε-greedy のパラメータ
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 1000000

# モデル保存ディレクトリ
CHECKPOINT_DIR = "checkpoints"

# ログ出力先
LOG_DIR = "logs"
