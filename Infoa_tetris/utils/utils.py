# utils/utils.py

import os
import datetime
import torch

def make_dir(directory):
    """
    ディレクトリが存在しなければ作成する
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model_checkpoint(agent, save_dir, episode):
    """
    エージェントのモデルをチェックポイントとして保存する。
    ファイル名にエピソード番号や日時を付与して管理すると便利。
    """
    make_dir(save_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"checkpoint_ep{episode}_{timestamp}.pth")
    torch.save(agent.policy_net.state_dict(), file_path)
    print(f"[INFO] Model saved to {file_path}")

def load_model_checkpoint(agent, model_path):
    """
    保存済みモデルを読み込む
    """
    agent.load(model_path)
    print(f"[INFO] Model loaded from {model_path}")