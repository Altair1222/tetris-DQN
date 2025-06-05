# env/tetris_env.py

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# テトリスのブロック（テトロミノ）を 4x4 のマトリクスで定義
# 各キー（0～6）は I, O, T, S, Z, J, L を表す
# 各リスト内に 4 回転分のマトリクスを格納
_TETROMINOES = {
    0: [  # I
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]], dtype=np.uint8),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=np.uint8),
    ],
    1: [  # O
        np.array([[0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
    ],
    2: [  # T
        np.array([[0, 1, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
    ],
    3: [  # S
        np.array([[0, 1, 1, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[1, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
    ],
    4: [  # Z
        np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 0, 0],
                  [1, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
    ],
    5: [  # J
        np.array([[1, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
    ],
    6: [  # L
        np.array([[0, 0, 1, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint8),
    ],
}


class TetrisEnv(gym.Env):
    """
    Gymnasium ベースの自作 Tetris 環境クラス。
    Observation: (height, width) の盤面マトリクス (0: 空, 1: ブロックあり)
    Action:
      0: 何もしない（No-op）
      1: 左に移動
      2: 右に移動
      3: 回転（90° 時計回り）
      4: ソフトドロップ（1 マス下へ移動）
      5: ハードドロップ（最下部まで一気に落とす）
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, height=20, width=10, render_mode=None, fig_width=8, fig_height=5):
        super(TetrisEnv, self).__init__()

        # --- パラメータ ---
        self.height = height  # 盤面の高さ
        self.width = width    # 盤面の幅
        self.render_mode = render_mode

        # 図の大きさ (インチ単位)。指定がなければ幅= self.width/2, 高さ= self.height/2 を使う
        if fig_width is None:
            self.fig_width = self.width / 2
        else:
            self.fig_width = fig_width
        if fig_height is None:
            self.fig_height = self.height / 2
        else:
            self.fig_height = fig_height

        # --- Observation / Action space の定義 ---
        # 盤面は 0: 空, 1-7: テトロミノID (I=1, O=2, ..., L=7) の uint8 行列
        self.observation_space = spaces.Box(
            low=0, high=7, shape=(self.height, self.width), dtype=np.uint8
        )
        # Action は Discrete(6)
        self.action_space = spaces.Discrete(6)

        # --- 内部状態 ---
        self.board = None                  # 盤面 (2D numpy array)
        self.current_piece = None          # 現在落下中のテトロミノ ID (0~6)
        self.current_rotation = None       # 回転インデックス (0~3)
        self.piece_x = None                # テトロミノの左上基準座標 (x)
        self.piece_y = None                # テトロミノの左上基準座標 (y)
        self.game_over = False             # ゲームオーバーフラグ
        self.score = 0                     # 累積スコア（ラインクリア数に比例）
        self.lines_cleared_total = 0       # 累積で消したライン数

        # Matplotlib 描画用キャッシュ
        self._fig = None
        self._ax = None
        self._im = None

        # 初期化
        self.reset()

    def reset(self, *, seed=None, options=None):
        """
        環境をリセットし、初期の観測を返す。
        """

        super().reset(seed=seed)

        # 盤面をすべてゼロで初期化
        self.board = np.zeros((self.height, self.width), dtype=np.uint8)

        # 新しいテトロミノを生成
        self._spawn_new_piece()

        # ゲームオーバーフラグとスコアをクリア
        self.game_over = False
        self.score = 0
        self.lines_cleared_total = 0

        # 初期観測値を返す
        observation = self._get_observation()
        info = {"score": self.score, "lines_cleared": 0}
        return observation, info

    def step(self, action):
        """
        1 ステップ分を進める。
        action: 0～5 の整数
        return: (observation, reward, terminated, truncated, info)
        """

        if self.game_over:
            # ゲームオーバー後は再度 reset されることを期待する。
            # ここでは同じ状態を返し、終了フラグを True にしておく。
            observation = self._get_observation()
            return observation, 0.0, True, False, {"score": self.score}

        reward = 0.0
        info = {}

        # --- 1) Action を適用 ---
        if action == 1:
            # 左に移動
            if self._valid_position(self.piece_x - 1, self.piece_y, self.current_rotation):
                self.piece_x -= 1

        elif action == 2:
            # 右に移動
            if self._valid_position(self.piece_x + 1, self.piece_y, self.current_rotation):
                self.piece_x += 1

        elif action == 3:
            # 回転（90度時計回り）
            new_rot = (self.current_rotation + 1) % 4
            if self._valid_position(self.piece_x, self.piece_y, new_rot):
                self.current_rotation = new_rot

        elif action == 4:
            # ソフトドロップ (1 マス下に移動)
            if self._valid_position(self.piece_x, self.piece_y + 1, self.current_rotation):
                self.piece_y += 1
            else:
                # これ以上下に行けない → 固定処理
                self._lock_piece()
                lines = self._clear_lines()
                reward += lines * 200.0
                self.lines_cleared_total += lines
                self.score += lines * 200.0
                self._spawn_new_piece()
                # 新ピースが置けない場合はゲームオーバー
                if not self._valid_position(self.piece_x, self.piece_y, self.current_rotation):
                    self.game_over = True
                else:
                    reward+=1
                

        elif action == 5:
            # ハードドロップ (可能な限り落とす)
            while self._valid_position(self.piece_x, self.piece_y + 1, self.current_rotation):
                self.piece_y += 1
            # 落ち切ったところで固定
            self._lock_piece()
            lines = self._clear_lines()
            reward += lines * 200.0
            self.lines_cleared_total += lines
            self.score += lines * 200.0
            self._spawn_new_piece()
            if not self._valid_position(self.piece_x, self.piece_y, self.current_rotation):
                self.game_over = True
            else:
                reward+=1

        # action == 0 の場合は「何もしない」
        # ただし、通常 gravity （重力落下）を適用する

        # --- 2) 重力落下 (何もしない or 回転/左右移動 の後に) ---
        if action in {0, 1, 2, 3}:
            # ソフトドロップ／ハードドロップは既に落下 or 固定処理しているのでスキップ
            if self._valid_position(self.piece_x, self.piece_y + 1, self.current_rotation):
                self.piece_y += 1
            else:
                # これ以上下に行けない → 固定処理
                self._lock_piece()
                lines = self._clear_lines()
                reward += lines * 200.0
                self.lines_cleared_total += lines
                self.score += lines * 200.0
                self._spawn_new_piece()
                if not self._valid_position(self.piece_x, self.piece_y, self.current_rotation):
                    self.game_over = True
                else:
                    reward+=1

        # --- 3) 終了判定 ---
        terminated = self.game_over
        truncated = False  # ステップ数制限などは導入していないので常に False

        # --- 4) 次の観測 ---
        observation = self._get_observation()
        info = {"score": self.score, "lines_cleared": self.lines_cleared_total}
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        環境の可視化 (テキスト表示または画像表示)
        human: Matplotlib で盤面を表示
        rgb_array: (height*cell_size, width*cell_size, 3) の numpy array を返す（簡易実装）
        """
        if mode == "human":
            obs = self._get_observation()

            # --- 色マップと正規化 ---
            import matplotlib.pyplot as plt
            from matplotlib import colors
            from matplotlib.patches import Rectangle
            cmap = colors.ListedColormap(
                ["white", "cyan", "yellow", "purple", "green", "red", "blue", "orange"]
            )
            norm = colors.BoundaryNorm(boundaries=[-0.5] + [i + 0.5 for i in range(8)], ncolors=8)

            # 初回呼び出し時に Figure と Axes、Image を作成
            if self._fig is None or self._ax is None or self._im is None:
                plt.ion()
                self._fig, self._ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
                # 図のサイズをプロパティに合わせて固定（以後変更可能）
                self._fig.set_size_inches(self.fig_width, self.fig_height, forward=True)
                self._im = self._ax.imshow(obs, cmap=cmap, norm=norm, interpolation="nearest")
                self._ax.set_title(f"Score: {self.score}")
                self._ax.axis("off")
                # 枠線を描画
                rect = Rectangle(
                    (-0.5, -0.5), self.width, self.height,
                    fill=False, edgecolor="black", linewidth=2
                )
                self._ax.add_patch(rect)
                plt.show(block=False)
            else:
                # 図のサイズを現時点のプロパティに合わせる
                self._fig.set_size_inches(self.fig_width, self.fig_height, forward=True)
                # 既存の Image オブジェクトにデータを流し込んで更新
                self._im.set_data(obs)
                self._ax.set_title(f"Score: {self.score}")

            # 描画を更新して、短時間ポーズを入れる
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)

        elif mode == "rgb_array":
            # 簡易的に、盤面を 10x10 ピクセルのセルで描画し RGB array を返す
            cell_size = 10
            img = np.zeros((self.height * cell_size, self.width * cell_size, 3), dtype=np.uint8)
            obs = self._get_observation()
            for i in range(self.height):
                for j in range(self.width):
                    if obs[i, j] == 1:
                        img[
                            i * cell_size:(i + 1) * cell_size,
                            j * cell_size:(j + 1) * cell_size,
                            :
                        ] = np.array([255, 255, 255], dtype=np.uint8)
            return img

        else:
            raise NotImplementedError(f"Unknown render mode: {mode}")

    def close(self):
        """
        環境終了時のクリーンアップ (今回は特に何もしない)
        """
        pass

    # ======================
    # --- 内部ヘルパー関数 ---
    # ======================

    def _spawn_new_piece(self):
        """
        新しいテトロミノをランダムに生成し、初期位置にセットする。
        初期位置: ボード上部中央 (x = width//2 - 2, y = 0)
        """
        self.current_piece = self.np_random.integers(low=0, high=len(_TETROMINOES))
        self.current_rotation = 0
        # 4x4 マトリクスを基準にするので、x は (width//2 - 2)
        self.piece_x = (self.width // 2) - 2
        self.piece_y = 0

    def _get_observation(self):
        """
        盤面 (固定済みブロック) と現在のテトロミノを重ねた観測行列を返す。
        return: shape = (height, width), dtype=np.uint8
        """
        obs = self.board.copy()

        # 現在ブロックを描画
        shape = _TETROMINOES[self.current_piece][self.current_rotation]
        for i in range(4):
            for j in range(4):
                if shape[i, j]:
                    board_y = self.piece_y + i
                    board_x = self.piece_x + j
                    if 0 <= board_x < self.width and 0 <= board_y < self.height:
                        obs[board_y, board_x] = self.current_piece + 1
        return obs

    def _valid_position(self, x, y, rotation):
        """
        (x, y, rotation) の位置にテトロミノを置いたときに
        衝突や盤面外にはみ出しがないかチェックする。
        return: True (有効) / False (衝突 or はみ出し)
        """
        shape = _TETROMINOES[self.current_piece][rotation]
        for i in range(4):
            for j in range(4):
                if shape[i, j]:
                    board_y = y + i
                    board_x = x + j
                    # 盤面外 or すでに固定ブロックがある場合は無効
                    if board_x < 0 or board_x >= self.width or board_y < 0 or board_y >= self.height:
                        return False
                    if self.board[board_y, board_x]:
                        return False
        return True

    def _lock_piece(self):
        """
        現在のテトロミノを盤面に固定する (board に テトロミノID+1 を書き込む)。
        """
        shape = _TETROMINOES[self.current_piece][self.current_rotation]
        for i in range(4):
            for j in range(4):
                if shape[i, j]:
                    board_y = self.piece_y + i
                    board_x = self.piece_x + j
                    if 0 <= board_y < self.height and 0 <= board_x < self.width:
                        self.board[board_y, board_x] = self.current_piece + 1

    def _clear_lines(self):
        """
        盤面上の揃ったラインを消す。
        消したライン数を返し、盤面上部に空行を追加する。
        """
        lines_cleared = 0
        new_board = []

        for row in range(self.height):
            if np.all(self.board[row, :] == 1):
                lines_cleared += 1
            else:
                new_board.append(self.board[row, :])

        # 消去された行数分だけ上部にゼロ行を挿入
        if lines_cleared > 0:
            empty_rows = [np.zeros((self.width,), dtype=np.uint8) for _ in range(lines_cleared)]
            new_board = empty_rows + new_board

        # 行が少なくなった場合は上部に追加
        while len(new_board) < self.height:
            new_board.insert(0, np.zeros((self.width,), dtype=np.uint8))

        self.board = np.vstack(new_board)
        return lines_cleared