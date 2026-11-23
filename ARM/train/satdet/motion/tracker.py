"""多目标跟踪：卡尔曼预测 + 匈牙利关联，输出稳定 ID 与速度状态。

为什么不用 IoU 做关联（SORT/ByteTrack 的默认做法）
------------------------------------------------
目标是 6 px、帧间位移 2.5 px，相邻帧的 IoU 只有 0.3~0.5，且抖动极大；
用 IoU 阈值做门控会频繁丢关联。本项目改用**归一化中心距 + 尺寸相似度**：

    cost = d_center / max(4, 0.5*(w_gt + h_gt))  +  0.5 * |log(w_pred/w_gt)|

再配合卡尔曼预测（吸收匀速运动），在卫星小目标上比纯 IoU 稳得多。

输出字段（M6 的车速显示要用）
------------------------------
``TrackState.vx / vy``  px/frame，直接来自卡尔曼速度状态；
``age / hits / time_since_update`` 用于速度置信度判定（轨迹太新不给速度）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from satdet.core.logging import get_logger

__all__ = ["TrackState", "MultiObjectTracker", "TrackerConfig"]

log = get_logger("tracker")


@dataclass
class TrackerConfig:
    """跟踪参数。"""

    max_age: int = 15           # 计划2 条目6：8 -> 15，允许短暂遮挡（需大于 min_hits）
    min_hits: int = 3           # 计划2 条目9：2 -> 3，单帧噪点形不成轨迹
    gate_scale: float = 3.0     # 关联门限 = gate_scale × 目标尺度
    gate_min_px: float = 10.0   # 门限下限（6 px 目标不能太小）
    gate_norm_max: float = 1.0  # 归一化门限上限（norm=1 即椭圆/圆形边界；1.6 会伸到邻车道）
    size_weight: float = 0.5    # 尺寸差异在代价里的权重
    vel_weight: float = 0.5     # 速度一致性在代价里的权重

    # ---- 计划2 条目5：椭圆动态搜索域 + 唯一强制匹配 ----
    ellipse_along: float = 4.5      # 速度方向上的门限倍数（越大越宽容）
    ellipse_across: float = 2.2     # 垂直速度方向的倍数
    force_unique: bool = False      # 实测在密集场景有害（同一轨迹的候选常是邻居车），默认关闭

    # ---- 计划2 条目6：轨迹复活（第三轮：窗口真正生效 + 速度外推）----
    revive_window: int = 12         # 轨迹失效后多少帧内可被复活（超过即永久删除）
    revive_dist_px: float = 12.0    # 复活所需的预测位置距离
    revive_vel_decay: float = 0.7   # 复活时继承旧速度的比例（避免速度从 0 重新爬升）
    use_ellipse: bool = True        # 椭圆动态搜索域开关（便于消融）
    use_revive: bool = True         # 轨迹复活开关（便于消融）
    score_weight: float = 0.3       # 分数在关联代价里的权重（抑制误检抢轨迹）

    # ---- 第三轮：轨迹级重复抑制（不做框级去重！）----
    # 框级去重会把相邻车道的真车合并（实测 mAP -7.6~-9.1）；
    # 轨迹级要求"连续多帧位置、尺寸、速度都几乎一致"，才判定为同一辆车的影子轨迹。
    # 实现上是**抑制**（不输出、不领号）而不是删除：删除后下一帧同一检测会
    # 再生成一条新轨迹，影子 ID 会无限"再生"。
    dedup_tracks: bool = True
    dup_axis_ratio: float = 0.35    # 中心差 <= ratio × 较小边长（6 px 车 -> 2.1 px）
    dup_size_ratio: float = 1.5     # 宽/高之比上限
    dup_patience: int = 2           # 已转正轨迹连续多少帧重合才抑制较弱的一条（消融：2 优于 3）
    dup_patience_tentative: int = 1 # 两条都还没转正时立即抑制（同帧同位置的框不可能两辆车）


class _Kalman:
    """8 维状态 [cx, cy, w, h, vx, vy, vw, vh] 的匀速模型（纯 numpy）。"""

    def __init__(self) -> None:
        self.F = np.eye(8)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0
        self.H = np.eye(4, 8)
        self.Q = np.eye(8) * 1e-2
        self.R = np.eye(4) * 1e-1
        self.P = np.eye(8) * 10.0
        self.x = np.zeros((8, 1))

    def init(self, box: np.ndarray) -> None:
        cx, cy, w, h = _to_cxcywh(box)
        self.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]])
        self.P = np.eye(8) * 10.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _from_cxcywh(self.x[:4, 0])

    def update(self, box: np.ndarray) -> None:
        z = np.asarray(_to_cxcywh(box), dtype=np.float64).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return _from_cxcywh(self.x[:4, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])


def _to_cxcywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)


def _from_cxcywh(s: np.ndarray) -> np.ndarray:
    cx, cy, w, h = s
    w = max(w, 1.0)
    h = max(h, 1.0)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


@dataclass
class TrackState:
    """一条轨迹的对外状态。"""

    track_id: int
    box: np.ndarray
    vx: float
    vy: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    score: float = 0.0

    @property
    def speed_px(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "x1": float(self.box[0]), "y1": float(self.box[1]),
            "x2": float(self.box[2]), "y2": float(self.box[3]),
            "vx": self.vx, "vy": self.vy, "speed_px": self.speed_px,
            "age": self.age, "hits": self.hits,
            "time_since_update": self.time_since_update, "score": self.score,
        }


class _Track:
    """单条轨迹。

    **暂定态（tentative）**：新轨迹先不给正式 ID，连续命中 ``min_hits`` 帧后
    才"转正"并领号。这样单帧误检永远不会消耗 ID，也就不会制造 ID 跳变
    ——这是第二轮把端到端 IDSW 从 393 压下来的关键改动（详见 docs/15）。
    """

    __slots__ = ("state", "kf", "age", "hits", "tsu", "public_id",
                 "uid", "suppressed", "dup_release")

    def __init__(self, box: np.ndarray, score: float, uid: int = 0) -> None:
        self.public_id: int | None = None
        self.kf = _Kalman()
        self.kf.init(box)
        self.state = TrackState(-1, box.astype(np.float32), 0.0, 0.0,
                                age=0, hits=1, time_since_update=0, score=score)
        self.age = 1
        self.hits = 1
        self.tsu = 0
        self.uid = uid              # 内部唯一号（暂定态还没有 public_id，用它做去重键）
        self.suppressed = False     # 被判为影子轨迹：继续跟踪但不输出、不领号
        self.dup_release = 0        # 已不再与正常轨迹重合的连续帧数

    @property
    def confirmed(self) -> bool:
        return self.public_id is not None

    def promote(self, track_id: int) -> None:
        self.public_id = track_id
        self.state.track_id = track_id

    def predict(self) -> None:
        self.state.box = self.kf.predict()
        self.state.vx, self.state.vy = self.kf.velocity
        self.age += 1
        self.tsu += 1

    @property
    def velocity(self) -> tuple[float, float]:
        return self.kf.velocity

    def update(self, box: np.ndarray, score: float) -> None:
        self.kf.update(box)
        self.state.box = self.kf.box
        self.state.vx, self.state.vy = self.kf.velocity
        self.state.score = score
        self.hits += 1
        self.tsu = 0
        self.state.age = self.age
        self.state.hits = self.hits
        self.state.time_since_update = 0


@dataclass
class MultiObjectTracker:
    """检测结果 -> 轨迹。``update()`` 返回本帧**已转正**的轨迹列表。

    第二轮按计划2 条目 5/6/9 强化，每条都做过消融（docs/15）：

    * **暂定态 + 转正才领号**：单帧误检不消耗 ID（IDSW 主因）；
    * **椭圆动态搜索域**：沿速度方向放大门限（速度越快、候选区越长）；
    * **轨迹复活**：失效 ``revive_window`` 帧内若新检测落在旧轨迹预测位置附近，
      复用原 ID；
    * **唯一强制匹配**：默认关闭（密集场景实测有害，IDSW 43 -> 122）；
    * **分数加权关联**：低分检测的代价更高，避免误检抢走正确轨迹。
    """

    cfg: TrackerConfig = field(default_factory=TrackerConfig)
    _tracks: list[_Track] = field(default_factory=list, repr=False)
    # (track_id, 死亡时框, vx, vy, 死亡帧号, 分数)
    _dead: list[tuple[int, np.ndarray, float, float, int, float]] = field(
        default_factory=list, repr=False)
    _next_id: int = 1
    _next_uid: int = 1
    _frame: int = 0
    _dup_streak: dict[tuple[int, int], int] = field(default_factory=dict, repr=False)

    def reset(self) -> None:
        self._tracks = []
        self._dead = []
        self._next_id = 1
        self._next_uid = 1
        self._frame = 0
        self._dup_streak = {}

    def _new_track(self, box: np.ndarray, score: float) -> "_Track":
        tr = _Track(box, score, self._next_uid)
        self._next_uid += 1
        return tr

    # ------------------------------------------------------------------
    def update(self, det_boxes: np.ndarray, det_scores: np.ndarray | None = None
               ) -> list[TrackState]:
        """推进一帧。``det_boxes`` 为 (N,4) xyxy；``det_scores`` 可选。"""
        self._frame += 1
        boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 4)
        scores = (np.ones(len(boxes), np.float32) if det_scores is None
                  else np.asarray(det_scores, np.float32).reshape(-1))

        # 1) 全部轨迹先预测
        for tr in self._tracks:
            tr.predict()

        # 2) 关联（椭圆门限 + 分数加权）
        matches, unmatched_dets = self._associate(boxes, scores)
        for ti, di in matches:
            self._tracks[ti].update(boxes[di], float(scores[di]))

        # 3) 未命中的检测：先尝试复活旧轨迹，再新建（暂定态，不领号）
        still: list[int] = []
        for di in unmatched_dets:
            rev = self.cfg.use_revive and self._try_revive(boxes[di], float(scores[di]))
            if not rev:
                still.append(di)
        for di in still:
            self._tracks.append(self._new_track(boxes[di], float(scores[di])))

        # 4) 轨迹级重复抑制：**必须在转正之前**。两条暂定轨迹同帧重合时，
        #    较弱的一条直接抑制，连 ID 都不会领——否则影子 ID 会在前几帧先输出一次，
        #    给评测平白制造一次 IDSW。
        if self.cfg.dedup_tracks:
            self._suppress_duplicates()

        # 5) 转正：命中够数才领正式 ID（被抑制的影子轨迹不领号）
        for tr in self._tracks:
            if tr.public_id is None and not tr.suppressed and tr.hits >= self.cfg.min_hits:
                tr.promote(self._next_id)
                self._next_id += 1

        # 6) 淘汰过期轨迹；已转正的进复活池
        alive: list[_Track] = []
        for tr in self._tracks:
            if tr.tsu > self.cfg.max_age:
                # 影子轨迹不进复活池，否则它会在别处"借尸还魂"
                if tr.confirmed and not tr.suppressed:
                    vx, vy = tr.velocity
                    self._dead.append((int(tr.public_id), tr.state.box.copy(),
                                       float(vx), float(vy), self._frame,
                                       float(tr.state.score)))
            else:
                alive.append(tr)
        self._tracks = alive
        # 复活池必须按时间过期：否则几十帧后同一车道上的新车会继承旧 ID。
        if self._dead:
            self._dead = [d for d in self._dead
                          if self._frame - d[4] <= self.cfg.revive_window][-64:]

        # 7) 输出：只输出已转正、本帧有更新、且不是影子的轨迹
        out: list[TrackState] = []
        for tr in self._tracks:
            if tr.tsu > 0 or not tr.confirmed or tr.suppressed:
                continue
            tr.state.time_since_update = 0
            out.append(tr.state)
        return out

    # ------------------------------------------------------------------
    def _try_revive(self, box: np.ndarray, score: float) -> bool:
        """若检测落在某条**未过期**失效轨迹的外推位置附近，复活它并复用 ID。

        第三轮修正了两个问题：

        1. ``revive_window`` 以前只是配置里的一个数字，复活时完全不看失效时长，
           几十帧后的新车也会继承旧 ID；现在按帧号严格过期。
        2. 以前拿"死亡瞬间的框"比距离，快车 2.5 px/帧时 12 px 半径只够 5 帧。
           现在先用旧速度外推到当前帧，再比距离 —— 窗口与距离两个参数才自洽。
        """
        if not self._dead:
            return False
        c = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        best, best_d = -1, float(self.cfg.revive_dist_px)
        for k, (_tid, dbox, vx, vy, dframe, _sc) in enumerate(self._dead):
            age = self._frame - int(dframe)
            if age <= 0 or age > self.cfg.revive_window:
                continue
            dc = np.asarray([(dbox[0] + dbox[2]) / 2, (dbox[1] + dbox[3]) / 2])
            dc = dc + np.asarray([vx, vy]) * float(age)      # 匀速外推
            d = float(np.linalg.norm(c - dc))
            if d < best_d:
                best, best_d = k, d
        if best < 0:
            return False
        tid, _dbox, vx, vy, _dframe, _sc = self._dead.pop(best)
        tr = self._new_track(box, score)
        tr.promote(tid)          # 复活即转正，直接复用原 ID
        tr.hits = max(tr.hits, self.cfg.min_hits)
        # 继承旧速度：否则复活后速度从 0 重新爬升，会制造一段假低速并放大限幅滞后。
        decay = float(self.cfg.revive_vel_decay)
        tr.kf.x[4, 0] = vx * decay
        tr.kf.x[5, 0] = vy * decay
        tr.state.vx, tr.state.vy = vx * decay, vy * decay
        self._tracks.append(tr)
        return True

    # ------------------------------------------------------------------
    def _suppress_duplicates(self) -> None:
        """抑制"同一辆车被两条轨迹同时跟踪"的影子轨迹。

        与框级去重的区别：这里只要求**两条本帧刚更新的轨迹**在位置/尺寸/速度上
        三重一致，且连续满足。相邻车道的车中心差约一个车道宽（~4 px），
        不满足"两轴都在 0.35 倍边长内"，因此不会被合并。

        处理方式是**抑制而非删除**：影子轨迹继续跟踪、继续吃自己的检测，
        只是不领号、不输出。删除会让下一帧同一检测立刻生成新轨迹，
        影子 ID 每 ``dup_patience`` 帧再生一次，IDSW 反而爆炸。
        """
        cand = [tr for tr in self._tracks if tr.tsu == 0 and not tr.suppressed]
        new_streak: dict[tuple[int, int], int] = {}
        if len(cand) >= 2:
            for i, a in enumerate(cand):
                if a.suppressed:
                    continue
                for b in cand[i + 1:]:
                    if b.suppressed or not _is_duplicate_track(a, b, self.cfg):
                        continue
                    key = (min(a.uid, b.uid), max(a.uid, b.uid))
                    n = self._dup_streak.get(key, 0) + 1
                    new_streak[key] = n
                    # 两条都没转正：同帧同位置不可能是两辆车，立刻抑制
                    need = (self.cfg.dup_patience if (a.confirmed and b.confirmed)
                            else self.cfg.dup_patience_tentative)
                    if n >= need:
                        # 保留更"老"的（hits 多），同 hits 时保留分数高的
                        weak = b if (a.hits, a.state.score) >= (b.hits, b.state.score) else a
                        weak.suppressed = True
                        log.debug("去重：抑制影子轨迹 uid=%d/id=%s（与 uid=%d 连续 %d 帧重合）",
                                  weak.uid, weak.public_id, a.uid if weak is b else b.uid, n)
                        if weak is a:
                            break
        self._dup_streak = new_streak

        # 释放：影子轨迹与任何正常轨迹都不再重合。连续 ``dup_patience`` 帧（或
        # 本帧没被关联上）后恢复输出——两车交叉后分开的情况靠这一步救回来。
        active = [tr for tr in self._tracks if tr.tsu == 0 and not tr.suppressed]
        for tr in self._tracks:
            if not tr.suppressed:
                continue
            still_shadow = any(_is_duplicate_track(tr, a, self.cfg) for a in active)
            if still_shadow:
                tr.dup_release = 0
            else:
                tr.dup_release += 1
                if tr.tsu > 0 or tr.dup_release >= self.cfg.dup_patience:
                    tr.suppressed = False
                    tr.dup_release = 0

    # ------------------------------------------------------------------
    def _associate(self, boxes: np.ndarray, scores: np.ndarray
                   ) -> tuple[list[tuple[int, int]], list[int]]:
        n_tr, n_det = len(self._tracks), len(boxes)
        if n_tr == 0 or n_det == 0:
            return [], list(range(n_det))

        pred = np.stack([t.state.box for t in self._tracks], 0)          # (T,4)
        vel = np.stack([[t.state.vx, t.state.vy] for t in self._tracks], 0)
        cost = _association_cost(pred, boxes, self.cfg, vel, scores)
        big = 1e6
        cost = np.where(np.isnan(cost), big, cost)

        forced: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        if self.cfg.force_unique:
            valid = cost < big / 2
            for ti in range(n_tr):
                cand = np.flatnonzero(valid[ti])
                if cand.size == 1 and int(cand[0]) not in used_d:
                    forced.append((ti, int(cand[0])))
                    used_t.add(ti)
                    used_d.add(int(cand[0]))
            for di in range(n_det):
                if di in used_d:
                    continue
                cand = np.flatnonzero(valid[:, di])
                if cand.size == 1 and int(cand[0]) not in used_t:
                    forced.append((int(cand[0]), di))
                    used_t.add(int(cand[0]))
                    used_d.add(di)

        rest_t = [i for i in range(n_tr) if i not in used_t]
        rest_d = [i for i in range(n_det) if i not in used_d]
        matches = list(forced)
        if rest_t and rest_d:
            sub = cost[np.ix_(rest_t, rest_d)]
            rows, cols = linear_sum_assignment(sub)
            for r, c in zip(rows, cols):
                if sub[r, c] >= big / 2:
                    continue
                matches.append((rest_t[r], rest_d[c]))
                used_d.add(rest_d[c])

        unmatched = [i for i in range(n_det) if i not in used_d]
        return matches, unmatched


def _association_cost(pred: np.ndarray, det: np.ndarray, cfg: TrackerConfig,
                      vel: np.ndarray | None = None,
                      det_scores: np.ndarray | None = None) -> np.ndarray:
    """椭圆动态搜索域 + 尺寸相似度 + 分数加权的关联代价；超出搜索域置 NaN。

    椭圆沿**速度方向**拉长（速度越快，目标下一帧可能出现的范围越大），
    垂直方向收紧；速度接近 0 时退化为圆形门限（否则静止目标会被连断）。
    """
    pc = np.stack([(pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2], 1)
    dc = np.stack([(det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2], 1)
    pw = np.maximum(pred[:, 2] - pred[:, 0], 1.0)
    ph = np.maximum(pred[:, 3] - pred[:, 1], 1.0)
    dw = np.maximum(det[:, 2] - det[:, 0], 1.0)
    dh = np.maximum(det[:, 3] - det[:, 1], 1.0)

    delta = dc[None, :, :] - pc[:, None, :]
    dist = np.linalg.norm(delta, axis=2)
    base = np.maximum(0.5 * (pw + ph) / 2.0, 3.0)

    if vel is not None and np.any(vel) and cfg.use_ellipse:
        v = vel.astype(np.float64)
        speed = np.linalg.norm(v, axis=1)
        safe = np.maximum(speed[:, None], 1e-6)
        along = np.where(speed[:, None] > 1e-6, (delta * v[:, None, :]).sum(-1) / safe, 0.0)
        across = np.sqrt(np.maximum(dist ** 2 - along ** 2, 0.0))
        a = base[:, None] * (cfg.ellipse_along + 0.6 * speed[:, None])
        b = base[:, None] * cfg.ellipse_across
        norm_ellipse = np.sqrt((along / a) ** 2 + (across / b) ** 2)
        gate = np.maximum(cfg.gate_min_px, cfg.gate_scale * base)[:, None]
        norm_circle = dist / gate
        norm = np.where(speed[:, None] > 1.0, norm_ellipse, norm_circle)
    else:
        gate = np.maximum(cfg.gate_min_px, cfg.gate_scale * base)[:, None]
        norm = dist / gate

    cost = norm + cfg.size_weight * (0.5 * (np.abs(np.log(pw[:, None] / dw[None, :]))
                                            + np.abs(np.log(ph[:, None] / dh[None, :]))))
    if det_scores is not None and cfg.score_weight > 0:
        # 低分检测代价更高：误检通常分数低，避免它把高分轨迹抢走
        cost = cost + cfg.score_weight * (1.0 - np.clip(det_scores, 0.0, 1.0))[None, :]
    # 归一化门限：norm=1 已经是椭圆/圆形边界。旧版硬编码 1.6 等于再放宽 60%，
    # 椭圆横轴会伸到邻车道，是端到端 IDSW 的主要来源之一。
    # 消融（002/003，120 帧）：1.6 -> 1.0 使 002 IDSW 92->52、003 70->48。
    cost = np.where(norm > cfg.gate_norm_max, np.nan, cost)
    return cost


def _is_duplicate_track(a: "_Track", b: "_Track", cfg: TrackerConfig) -> bool:
    """两条轨迹是否在跟踪同一辆车（位置、尺寸、速度三重一致性）。"""
    ba, bb = a.state.box, b.state.box
    wa = max(float(ba[2] - ba[0]), 1.0)
    ha = max(float(ba[3] - ba[1]), 1.0)
    wb = max(float(bb[2] - bb[0]), 1.0)
    hb = max(float(bb[3] - bb[1]), 1.0)

    acx, acy = (ba[0] + ba[2]) * 0.5, (ba[1] + ba[3]) * 0.5
    bcx, bcy = (bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5
    if abs(acx - bcx) > cfg.dup_axis_ratio * min(wa, wb):
        return False
    if abs(acy - bcy) > cfg.dup_axis_ratio * min(ha, hb):
        return False
    if max(wa, wb) / min(wa, wb) > cfg.dup_size_ratio:
        return False
    if max(ha, hb) / min(ha, hb) > cfg.dup_size_ratio:
        return False

    va = np.asarray([a.state.vx, a.state.vy], np.float64)
    vb = np.asarray([b.state.vx, b.state.vy], np.float64)
    sa, sb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if sa > 1.0 or sb > 1.0:      # 静止目标不比较方向（方向无意义）
        if sa * sb <= 1e-6:
            return False
        if float(va @ vb / (sa * sb)) < 0.9:
            return False
        if abs(sa - sb) > max(1.0, 0.5 * max(sa, sb)):
            return False
    return True
