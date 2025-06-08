#!/usr/bin/env python3
# human_play.py
# 人間がキー入力で遊べる TetrisEnv のプレイスクリプト

import sys
import time
from env.tetris_env import TetrisEnv

# キー → 環境アクションのマッピング
KEY_ACTION = {
    'a': 1,    # 左
    'd': 2,    # 右
    'w': 3,    # 回転
    's': 4,    # ソフトドロップ
    ' ': 5,    # ハードドロップ (スペースキー)
    '': 0      # 何もしない
}

def print_instructions():
    print("Human Play Tetris")
    print("-----------------")
    print("キー入力で操作します。")
    print("  a : 左移動")
    print("  d : 右移動")
    print("  w : 回転")
    print("  s : ソフトドロップ (1マス下)")
    print(" space : ハードドロップ")
    print("  q : 終了")
    print("Enterだけ押すと何もしません。")
    print()

def main():
    env = TetrisEnv(render_mode="human")
    state, info = env.reset()
    done = False

    print_instructions()

    while not done:
        # 画面表示
        env.render()

        # スコア表示
        print(f"Score: {info['score']}  Lines Cleared: {info['lines_cleared']}")
        # 入力受付
        key = input("Action [a/d/w/s/space/q]: ").lower()
        if key == 'q':
            print("Quitting...")
            break

        # キーからアクション取得（該当なければ 0=No-op）
        action = KEY_ACTION.get(key, 0)

        # 環境を 1 ステップ進める
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 少しだけウェイトを入れる（画面が早すぎる場合）
        time.sleep(0.05)

    print("Game Over! Final Score:", info.get('score', 0))
    env.close()

if __name__ == "__main__":
    main()