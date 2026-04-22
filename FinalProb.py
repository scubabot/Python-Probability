# FinalProb_updated.py
# Updated paper-closer version:
# - offline meeting selection from communication subareas
# - robustness probability + transmissions via Monte Carlo (r,s)-robustness
# - actual meeting nodes selected inside winning subarea
# - online planning with global sensed set sset
# - sensing-location selection using covariance-based one-step MI equivalent
# - softer communication scaling and gentler edge inflation

import math
import heapq
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


# ============================================================
# 1) Grid graph / environment
# ============================================================

@dataclass
class GridGraph:
    nx: int
    ny: int
    xlim: Tuple[float, float] = (-10.0, 10.0)
    ylim: Tuple[float, float] = (-5.0, 5.0)

    def __post_init__(self):
        self.nodes = [(ix, iy) for iy in range(self.ny) for ix in range(self.nx)]
        self.coords = np.array([self.node_to_xy(n) for n in self.nodes], dtype=float)
        self.base_edge_cost = 1.0
        self.edge_weight = {}

        for n in self.nodes:
            for m in self.neighbors(n):
                self.edge_weight[(n, m)] = self.base_edge_cost

    def node_to_xy(self, node: Tuple[int, int]) -> np.ndarray:
        ix, iy = node
        x = self.xlim[0] + (self.xlim[1] - self.xlim[0]) * ix / max(1, self.nx - 1)
        y = self.ylim[0] + (self.ylim[1] - self.ylim[0]) * iy / max(1, self.ny - 1)
        return np.array([x, y], dtype=float)

    def neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        ix, iy = node
        cand = [(ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)]
        return [(jx, jy) for jx, jy in cand if 0 <= jx < self.nx and 0 <= jy < self.ny]

    def shortest_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        pq = [(0.0, start)]
        dist = {start: 0.0}
        prev = {}
        seen = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)

            if u == goal:
                break

            for v in self.neighbors(u):
                nd = d + self.edge_weight[(u, v)]
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if goal not in dist:
            return [start]

        path = [goal]
        cur = goal
        while cur != start:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    def path_cost(self, path: List[Tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0.0
        return float(sum(self.edge_weight[(path[i], path[i + 1])] for i in range(len(path) - 1)))

    def inflate_path_edges(self, path: List[Tuple[int, int]], alpha: float = 1.2):
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            self.edge_weight[e] = alpha * self.edge_weight[e]


# ============================================================
# 2) True hidden environment field
# ============================================================

def true_environment_value(x, y):
    return (
        2.2 * np.sin(0.55 * x)
        + 1.8 * np.cos(0.70 * y)
        + 1.3 * np.sin(0.18 * x * y)
        + 0.9 * np.cos(0.35 * (x + 0.6 * y))
        + 1.2 * np.sin(0.25 * x + 0.15 * y)
    )


# ============================================================
# 3) Communication field surrogate
# ============================================================

def true_comms_value(x, y):
    val = (
        12.0
        + 4.5 * np.sin(0.22 * x)
        + 3.5 * np.cos(0.33 * y)
        + 2.0 * np.sin(0.11 * x * y)
        + 1.5 * np.cos(0.18 * (x - 0.5 * y))
    )
    return max(val, 1.0)


# ============================================================
# 4) GP helpers
# ============================================================

def build_gp(random_state: int = 0) -> GaussianProcessRegressor:
    kernel = (
        C(1.0, (1e-6, 1e4))
        * RBF(length_scale=1.5, length_scale_bounds=(1e-3, 1e4))
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1e2))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=1,
        random_state=random_state,
    )


def build_comms_gp() -> GaussianProcessRegressor:
    kernel = (
        C(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=1.0, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=False,
    )


def extract_hypers(gp: GaussianProcessRegressor) -> Tuple[float, float, float]:
    k = gp.kernel_
    amp = float(k.k1.k1.constant_value)
    ls = float(k.k1.k2.length_scale)
    noi = float(k.k2.noise_level)
    return amp, ls, noi


def build_gp_from_hypers(amp: float, ls: float, noi: float) -> GaussianProcessRegressor:
    kernel = (
        C(amp, constant_value_bounds="fixed")
        * RBF(length_scale=ls, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=noi, noise_level_bounds="fixed")
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=False,
    )


# ============================================================
# 5) W-MSR helpers
# ============================================================

def wmsr_1d(vals: List[float], F: int) -> float:
    vals = sorted(float(v) for v in vals)
    n = len(vals)
    if n == 0:
        return 0.0
    if 2 * F >= n:
        return float(np.mean(vals))
    return float(np.mean(vals[F:n - F]))


def aggregate_hypers(hlist: List[Tuple[float, float, float]], mode="wmsr", F: int = 1):
    amps = [h[0] for h in hlist]
    lens = [h[1] for h in hlist]
    nois = [h[2] for h in hlist]

    if mode == "mean":
        return float(np.mean(amps)), float(np.mean(lens)), float(np.mean(nois))
    if mode == "wmsr":
        return wmsr_1d(amps, F), wmsr_1d(lens, F), wmsr_1d(nois, F)
    raise ValueError("mode must be 'mean' or 'wmsr'")


# ============================================================
# 6) Exact-style robustness
# ============================================================

def sample_graph_from_prob(prob_mat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(prob_mat.shape) < prob_mat).astype(int)


def all_disjoint_nonempty_pairs(n: int):
    pairs = []
    nodes = list(range(n))

    for k in range(2, n + 1):
        for p in combinations(nodes, k):
            p = list(p)
            for j in range(1, math.floor(len(p) / 2) + 1):
                subsets = list(combinations(p, j))
                if len(p) == 2 * j:
                    subsets = subsets[: len(subsets) // 2]
                for s1 in subsets:
                    s1 = list(s1)
                    s2 = [x for x in p if x not in s1]
                    pairs.append((s1, s2))

    return pairs


def check_robust(A: np.ndarray, s1: List[int], s2: List[int], r: int, s: int) -> bool:
    n = A.shape[0]

    outside_s1 = [i for i in range(n) if i not in s1]
    A1 = A[np.ix_(outside_s1, s1)]
    sr1 = int(np.sum(np.sum(A1, axis=0) >= r))

    outside_s2 = [i for i in range(n) if i not in s2]
    A2 = A[np.ix_(outside_s2, s2)]
    sr2 = int(np.sum(np.sum(A2, axis=0) >= r))

    return sr1 == len(s1) or sr2 == len(s2) or (sr1 + sr2) >= s


def det_robust(A: np.ndarray, r: int, s: int) -> bool:
    n = A.shape[0]
    pairs = all_disjoint_nonempty_pairs(n)
    for s1, s2 in pairs:
        if not check_robust(A, s1, s2, r, s):
            return False
    return True


def monte_prob_robust(prob_mat: np.ndarray, r: int, s: int, iterations: int,
                      rng: np.random.Generator) -> float:
    total_robust = 0
    for _ in range(iterations):
        sample = sample_graph_from_prob(prob_mat, rng)
        if det_robust(sample, r, s):
            total_robust += 1
    return total_robust / iterations


def transmissions(prob_mat: np.ndarray, r: int, s: int, rng: np.random.Generator,
                  max_rounds: int = 100) -> int:
    total_instance = np.zeros_like(prob_mat, dtype=int)
    num = 0
    while True:
        sample = sample_graph_from_prob(prob_mat, rng)
        total_instance = np.logical_or(total_instance, sample).astype(int)
        if det_robust(total_instance, r, s) or num >= max_rounds:
            return num + 1
        num += 1


# ============================================================
# 7) Area / subarea helpers
# ============================================================

def split_into_areas(graph: GridGraph, m: int) -> List[List[Tuple[int, int]]]:
    cols = np.array_split(np.arange(graph.nx), m)
    areas = []
    for col_group in cols:
        area_nodes = []
        for iy in range(graph.ny):
            for ix in col_group:
                area_nodes.append((int(ix), int(iy)))
        areas.append(area_nodes)
    return areas


def split_area_into_subareas(area_nodes: List[Tuple[int, int]], n_sub: int) -> List[List[Tuple[int, int]]]:
    ordered = sorted(area_nodes, key=lambda t: (t[0], t[1]))
    return [list(chunk) for chunk in np.array_split(ordered, n_sub)]


def subarea_center_xy(graph: GridGraph, subarea: List[Tuple[int, int]]) -> np.ndarray:
    pts = np.array([graph.node_to_xy(n) for n in subarea], dtype=float)
    return np.mean(pts, axis=0)


def meeting_nodes_in_subarea(subarea: List[Tuple[int, int]], n_robots: int,
                             rng: np.random.Generator) -> List[Tuple[int, int]]:
    if len(subarea) >= n_robots:
        idx = rng.choice(len(subarea), size=n_robots, replace=False)
    else:
        idx = rng.choice(len(subarea), size=n_robots, replace=True)
    return [subarea[i] for i in idx]


def meeting_positions_from_nodes(graph: GridGraph, nodes: List[Tuple[int, int]]) -> np.ndarray:
    return np.array([graph.node_to_xy(n) for n in nodes], dtype=float)


# ============================================================
# 8) Communication probability matrix
# ============================================================

def build_comms_prob_matrix(graph: GridGraph,
                            meet_nodes: List[Tuple[int, int]],
                            y_max: float = 16.0) -> np.ndarray:
    n = len(meet_nodes)
    meet_xy = meeting_positions_from_nodes(graph, meet_nodes)
    meet_y = np.array([true_comms_value(xy[0], xy[1]) for xy in meet_xy], dtype=float)

    gps = []
    for j in range(n):
        center = meet_xy[j]
        Xtrain = [center]
        ytrain = [meet_y[j]]

        offsets = np.array([
            [0.4, 0.0], [-0.4, 0.0], [0.0, 0.4], [0.0, -0.4],
            [0.3, 0.3], [-0.3, 0.3], [0.3, -0.3], [-0.3, -0.3]
        ])

        for off in offsets:
            pt = center + off
            pt[0] = np.clip(pt[0], graph.xlim[0], graph.xlim[1])
            pt[1] = np.clip(pt[1], graph.ylim[0], graph.ylim[1])
            Xtrain.append(pt.copy())
            ytrain.append(true_comms_value(pt[0], pt[1]))

        gp = build_comms_gp()
        gp.fit(np.array(Xtrain, dtype=float), np.array(ytrain, dtype=float))
        gps.append(gp)

    prob_mat = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            pred = float(gps[a].predict(meet_xy[b].reshape(1, -1))[0])
            prob_mat[a, b] = float(np.clip(pred / y_max, 0.05, 0.99))

    return prob_mat


def probability_of_resilience_exact(graph: GridGraph,
                                    meet_nodes: List[Tuple[int, int]],
                                    r: int,
                                    s: int,
                                    iterations: int = 100,
                                    transmission_trials: int = 30,
                                    rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng(0)

    prob_mat = build_comms_prob_matrix(graph, meet_nodes, y_max=16.0)
    pr = monte_prob_robust(prob_mat, r=r, s=s, iterations=iterations, rng=rng)
    tx = [transmissions(prob_mat, r=r, s=s, rng=rng) for _ in range(transmission_trials)]
    return float(pr), float(np.mean(tx)), prob_mat


# ============================================================
# 9) Robot state
# ============================================================

@dataclass
class Robot:
    rid: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    X: List[np.ndarray]
    y: List[float]
    gp: Optional[GaussianProcessRegressor] = None
    current_node: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.current_node is None:
            self.current_node = self.start


# ============================================================
# 10) Sensing / attack / GP fit
# ============================================================

def sense_node(graph: GridGraph, node: Tuple[int, int], rng: np.random.Generator,
               meas_noise_std: float = 0.25, attacked: bool = False, attack_scale: float = 8.0):
    xy = graph.node_to_xy(node)
    val = true_environment_value(xy[0], xy[1]) + rng.normal(0.0, meas_noise_std)
    if attacked:
        val = val + rng.uniform(-attack_scale, attack_scale)
    return xy, float(val)


def fit_robot_gp(robot: Robot, seed: int = 0):
    if len(robot.X) < 8:
        return None
    X = np.array(robot.X, dtype=float)
    y = np.array(robot.y, dtype=float)
    gp = build_gp(random_state=seed)
    gp.fit(X, y)
    robot.gp = gp
    return gp


# ============================================================
# 11) Offline meeting selection
# ============================================================

def choose_best_meeting_subarea_exact(graph: GridGraph,
                                      subareas: List[List[Tuple[int, int]]],
                                      n_robots: int,
                                      r: int,
                                      s: int,
                                      rng: np.random.Generator,
                                      iterations: int = 100,
                                      transmission_trials: int = 30):
    stats = []

    for idx, sb in enumerate(subareas):
        meet_nodes = meeting_nodes_in_subarea(sb, n_robots, rng)
        pr, avg_rounds, prob_mat = probability_of_resilience_exact(
            graph=graph,
            meet_nodes=meet_nodes,
            r=r,
            s=s,
            iterations=iterations,
            transmission_trials=transmission_trials,
            rng=rng,
        )
        stats.append({
            "idx": idx,
            "subarea": sb,
            "meeting_nodes": meet_nodes,
            "center_xy": subarea_center_xy(graph, sb),
            "Pr": pr,
            "avg_rounds": avg_rounds,
            "prob_mat": prob_mat,
        })

    best = max(stats, key=lambda d: (d["Pr"], -d["avg_rounds"]))
    return best, stats


# ============================================================
# 12) MI-based sensing selection from covariance
# ============================================================

def conditional_variance_score(cov: np.ndarray, selected_idx: List[int], candidate_idx: int,
                               regularize: float = 0.1) -> float:
    cov = cov + regularize * np.eye(cov.shape[0])

    if len(selected_idx) == 0:
        return float(cov[candidate_idx, candidate_idx])

    S = np.array(selected_idx, dtype=int)
    j = int(candidate_idx)

    K_SS = cov[np.ix_(S, S)]
    K_jS = cov[np.ix_([j], S)]
    K_Sj = cov[np.ix_(S, [j])]
    K_jj = cov[j, j]

    try:
        inv = np.linalg.pinv(K_SS)
        cond_var = float(K_jj - K_jS @ inv @ K_Sj)
    except Exception:
        cond_var = float(K_jj)

    return max(cond_var, 0.0)


def best_intermediate_node_mi(graph: GridGraph,
                              robot: Robot,
                              area_nodes: List[Tuple[int, int]],
                              sset: Set[Tuple[int, int]],
                              regularize: float = 0.1) -> Tuple[int, int]:
    if len(area_nodes) == 0:
        return robot.current_node

    if robot.gp is None:
        unsensed = [n for n in area_nodes if n not in sset]
        if unsensed:
            return unsensed[0]
        return area_nodes[0]

    Xarea = np.array([graph.node_to_xy(n) for n in area_nodes], dtype=float)
    _, cov_area = robot.gp.predict(Xarea, return_cov=True)

    area_index = {node: i for i, node in enumerate(area_nodes)}
    selected_in_area = [area_index[n] for n in area_nodes if n in sset]

    best_node = area_nodes[0]
    best_score = -1.0

    for cand in area_nodes:
        if cand in sset:
            continue
        j = area_index[cand]
        score = conditional_variance_score(cov_area, selected_in_area, j, regularize=regularize)
        if score > best_score:
            best_score = score
            best_node = cand

    if best_score < 0:
        return area_nodes[0]

    return best_node


def nearest_node_to_xy(graph: GridGraph, xy: np.ndarray) -> Tuple[int, int]:
    d = np.linalg.norm(graph.coords - xy.reshape(1, 2), axis=1)
    return graph.nodes[int(np.argmin(d))]


def plan_robot_path(graph: GridGraph,
                    robot: Robot,
                    area_nodes: List[Tuple[int, int]],
                    next_meeting_xy: np.ndarray,
                    sset: Set[Tuple[int, int]],
                    regularize: float = 0.1):
    s = robot.current_node
    q = best_intermediate_node_mi(graph, robot, area_nodes, sset, regularize=regularize)
    t = nearest_node_to_xy(graph, next_meeting_xy)

    p1 = graph.shortest_path(s, q)
    p2 = graph.shortest_path(q, t)

    if len(p1) > 0 and len(p2) > 0 and p1[-1] == p2[0]:
        full = p1 + p2[1:]
    else:
        full = p1 + p2

    return full, q, t


def trim_path_to_budget(graph: GridGraph, path: List[Tuple[int, int]], max_cost: float):
    if len(path) <= 1:
        return path, 0.0

    trimmed = [path[0]]
    running = 0.0

    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        edge_c = graph.edge_weight[e]
        if running + edge_c > max_cost:
            break
        trimmed.append(path[i + 1])
        running += edge_c

    return trimmed, float(running)


# ============================================================
# 13) Plot helpers
# ============================================================

def paper_background_field(n=31, xlim=(-10, 10), ylim=(-5, 5)):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = true_environment_value(XX, YY)
    return XX, YY, ZZ


def plot_paper_style_paths_5panel(outdir: Path, graph: GridGraph, all_stage_paths, chosen_meeting_nodes):
    XX, YY, Z = paper_background_field(n=41, xlim=graph.xlim, ylim=graph.ylim)

    fig = plt.figure(figsize=(19, 4.4), dpi=240)
    axes = [fig.add_subplot(1, 5, i + 1) for i in range(5)]

    def draw_common(ax):
        ax.pcolormesh(XX, YY, Z, shading="auto")
        ax.grid(True, linewidth=0.45, alpha=0.65)
        ax.set_xlim(graph.xlim)
        ax.set_ylim(graph.ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

        for meet_nodes in chosen_meeting_nodes[:-1]:
            pts = np.array([graph.node_to_xy(n) for n in meet_nodes], dtype=float)
            ax.scatter(pts[:, 0], pts[:, 1], c="red", s=20, marker="x")

    stage1 = all_stage_paths[0]["paths"]
    for r_idx in range(min(4, len(stage1))):
        ax = axes[r_idx]
        draw_common(ax)
        ax.set_title("Planning Paths", fontsize=11)

        pts = np.array([graph.node_to_xy(n) for n in stage1[r_idx]["path"]], dtype=float)
        if len(pts) > 0:
            ax.plot(pts[:, 0], pts[:, 1], lw=2.7, color="k")
            ax.scatter(pts[:, 0], pts[:, 1], s=8, c="red", alpha=0.75, marker=".")

    ax = axes[4]
    draw_common(ax)
    ax.set_title("Planning Paths", fontsize=11)

    if len(all_stage_paths) > 1:
        stage2 = all_stage_paths[1]["paths"]
        for info in stage2:
            pts = np.array([graph.node_to_xy(n) for n in info["path"]], dtype=float)
            if len(pts) > 0:
                ax.plot(pts[:, 0], pts[:, 1], lw=2.0)
                ax.scatter(pts[:, 0], pts[:, 1], s=6, c="red", alpha=0.35, marker=".")

    caps = [
        "(a) Path for robot 1",
        "(b) Path for robot 2",
        "(c) Path for robot 3",
        "(d) Path for robot 4",
        "(e) Re-planning after meeting",
    ]

    for i, cap in enumerate(caps):
        axes[i].text(0.5, -0.22, cap, transform=axes[i].transAxes,
                     ha="center", fontsize=11)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outdir / "paper_fig5_paths.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gp_surfaces_4panel_paper(outdir: Path, graph: GridGraph, robots: List[Robot],
                                  good_ids: List[int], wmsr_hypers, mean_hypers,
                                  init_points: int = 20, grid_n: int = 65):
    xs = np.linspace(graph.xlim[0], graph.xlim[1], grid_n)
    ys = np.linspace(graph.ylim[0], graph.ylim[1], grid_n)
    XX, YY = np.meshgrid(xs, ys)
    Xgrid = np.c_[XX.ravel(), YY.ravel()]

    Z_true = true_environment_value(XX, YY)

    X_init_list, y_init_list = [], []
    for i in good_ids:
        Xi = np.array(robots[i].X, dtype=float) if len(robots[i].X) > 0 else np.empty((0, 2))
        yi = np.array(robots[i].y, dtype=float) if len(robots[i].y) > 0 else np.empty((0,))
        if Xi.shape[0] > 0:
            X_init_list.append(Xi[:init_points])
            y_init_list.append(yi[:init_points])

    if len(X_init_list) > 0:
        X_init = np.vstack(X_init_list)
        y_init = np.concatenate(y_init_list)
    else:
        X_init = np.empty((0, 2))
        y_init = np.empty((0,))

    if X_init.shape[0] >= 8:
        gp_init = build_gp(random_state=999)
        gp_init.fit(X_init, y_init)
        mu_init = gp_init.predict(Xgrid).reshape(YY.shape)
    else:
        mu_init = np.zeros_like(YY)

    X_good_list = [np.array(robots[i].X, dtype=float) for i in good_ids if len(robots[i].X) > 0]
    y_good_list = [np.array(robots[i].y, dtype=float) for i in good_ids if len(robots[i].y) > 0]

    if len(X_good_list) > 0:
        X_good = np.vstack(X_good_list)
        y_good = np.concatenate(y_good_list)
    else:
        X_good = np.empty((0, 2))
        y_good = np.empty((0,))

    if X_good.shape[0] >= 10 and wmsr_hypers is not None:
        aw, lw, nw = wmsr_hypers
        gp_w = build_gp_from_hypers(aw, lw, nw)
        gp_w.fit(X_good, y_good)
        mu_w = gp_w.predict(Xgrid).reshape(YY.shape)
    else:
        mu_w = np.zeros_like(YY)

    X_all_list = [np.array(rb.X, dtype=float) for rb in robots if len(rb.X) > 0]
    y_all_list = [np.array(rb.y, dtype=float) for rb in robots if len(rb.y) > 0]

    if len(X_all_list) > 0:
        X_all = np.vstack(X_all_list)
        y_all = np.concatenate(y_all_list)
    else:
        X_all = np.empty((0, 2))
        y_all = np.empty((0,))

    if X_all.shape[0] >= 10 and mean_hypers is not None:
        am, lm, nm = mean_hypers
        gp_m = build_gp_from_hypers(am, lm, nm)
        gp_m.fit(X_all, y_all)
        mu_m = gp_m.predict(Xgrid).reshape(YY.shape)
    else:
        mu_m = np.zeros_like(YY)

    Zs = [Z_true, mu_init, mu_w, mu_m]
    zmin = min(float(np.min(Z)) for Z in Zs)
    zmax = max(float(np.max(Z)) for Z in Zs)

    fig = plt.figure(figsize=(18, 5.1), dpi=200)
    titles = ["Actual GP", "Initial GP", "Resilient MIPP", "Non-Resilient MIPP"]
    caps = [
        "(a) Actual environmental GP",
        "(b) Initial knowledge of the GP",
        "(c) Learned GP using resilient MIPP",
        "(d) Learned GP using non-resilient MIPP",
    ]

    for i, Z in enumerate(Zs):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.plot_surface(XX, YY, Z, linewidth=0.2, antialiased=True)
        ax.set_title(titles[i], pad=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlim(zmin, zmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=25, azim=235)

    for i, cap in enumerate(caps):
        fig.text((i + 0.5) / 4.0, 0.02, cap, ha="center", va="bottom", fontsize=12)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outdir / "paper_fig6_gp_4panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 5.6), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    diff = np.abs(mu_w - mu_m)
    ax.plot_surface(XX, YY, diff, linewidth=0.2, antialiased=True)
    ax.set_title("|Resilient GP - Non-Resilient GP|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=25, azim=235)
    plt.tight_layout()
    plt.savefig(outdir / "paper_fig6_difference.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 14) Main
# ============================================================

def main():
    OUTDIR = Path("paper_plots")
    OUTDIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)

    n_robots = 4
    F = 1
    r = 2
    s = 2

    m_areas = 3
    n_subareas_per_area = 10
    total_budget = 250.0
    stage_budget_per_robot = 25.0
    meas_noise_std = 0.25
    resilience_iterations = 100
    transmission_trials = 20
    regularize = 0.1

    graph = GridGraph(nx=25, ny=25, xlim=(-10, 10), ylim=(-5, 5))
    areas = split_into_areas(graph, m=m_areas)

    start_nodes = [(0, 3), (0, 8), (0, 13), (0, 18)]
    goal_nodes = [(24, 3), (24, 8), (24, 13), (24, 18)]

    robots = [Robot(rid=i, start=start_nodes[i], goal=goal_nodes[i], X=[], y=[]) for i in range(n_robots)]

    faulty_set = {n_robots - 1}
    BAD_H = (800.0, 25.0, 50.0)

    sset: Set[Tuple[int, int]] = set(start_nodes)

    print("\n=== OFFLINE STEP: choose meeting subareas ===")
    chosen_meeting_nodes = []

    for area_idx in range(m_areas - 1):
        subareas = split_area_into_subareas(areas[area_idx], n_subareas_per_area)

        best, stats = choose_best_meeting_subarea_exact(
            graph=graph,
            subareas=subareas,
            n_robots=n_robots,
            r=r,
            s=s,
            rng=rng,
            iterations=resilience_iterations,
            transmission_trials=transmission_trials,
        )

        chosen_meeting_nodes.append(best["meeting_nodes"])

        center = best["center_xy"]
        print(
            f"Area {area_idx}: best subarea idx={best['idx']}, "
            f"center={center}, Pr={best['Pr']:.3f}, avg_rounds={best['avg_rounds']:.2f}"
        )
        print(f"  chosen meeting nodes: {best['meeting_nodes']}")
        print("  Candidate subareas:")
        for st in stats:
            print(f"    sb={st['idx']:02d} | Pr={st['Pr']:.3f} | avg_rounds={st['avg_rounds']:.2f}")

    chosen_meeting_nodes.append(goal_nodes)

    print("\n=== ONLINE STEP: explore, meet, share hypers, replan ===")
    used_budget = 0.0
    all_stage_paths = []
    last_wmsr = None
    last_mean = None

    for stage_idx, area_nodes in enumerate(areas):
        next_meeting_nodes = chosen_meeting_nodes[stage_idx]
        print(f"\n--- Stage {stage_idx + 1}/{m_areas} ---")

        for rb in robots:
            fit_robot_gp(rb, seed=100 + rb.rid)

        planned_paths = []

        for k in range(n_robots):
            next_meeting_xy = graph.node_to_xy(next_meeting_nodes[k])
            path, q, t = plan_robot_path(
                graph, robots[k], area_nodes, next_meeting_xy, sset, regularize=regularize
            )

            raw_cost = graph.path_cost(path)
            remaining_global = max(0.0, total_budget - used_budget)
            allowed_cost = min(stage_budget_per_robot, remaining_global)

            if raw_cost > allowed_cost:
                path, c = trim_path_to_budget(graph, path, allowed_cost)
                print(f"Robot {k}: path trimmed to fit budget.")
            else:
                c = raw_cost

            if len(path) == 0:
                path = [robots[k].current_node]
                c = 0.0

            planned_paths.append({
                "robot": k,
                "path": path,
                "q": q,
                "t": t,
                "cost": c,
            })

            used_budget += c

            # update selected set like MATLAB sset = union(sset, p)
            for node in path:
                sset.add(node)

            # gentler path diversification after final accepted path
            graph.inflate_path_edges(path, alpha=1.2)

            print(
                f"Robot {k}: q={q}, t={t}, meet_node={next_meeting_nodes[k]}, "
                f"path_len={len(path)}, cost={c:.2f}"
            )

        all_stage_paths.append({
            "stage": stage_idx + 1,
            "paths": planned_paths,
            "meeting_nodes": next_meeting_nodes,
        })

        for info in planned_paths:
            rb = robots[info["robot"]]
            attacked = rb.rid in faulty_set

            for node in info["path"]:
                xy, val = sense_node(
                    graph, node, rng,
                    meas_noise_std=meas_noise_std,
                    attacked=attacked,
                    attack_scale=8.0
                )
                rb.X.append(xy)
                rb.y.append(val)

            rb.current_node = info["path"][-1]

        local_hypers = []
        for rb in robots:
            gp = fit_robot_gp(rb, seed=200 + rb.rid)
            if gp is None:
                local_hypers.append((1.0, 1.5, 0.05))
            else:
                local_hypers.append(extract_hypers(gp))

        reported = []
        for rb in robots:
            if rb.rid in faulty_set:
                reported.append(BAD_H)
            else:
                reported.append(local_hypers[rb.rid])

        last_wmsr = aggregate_hypers(reported, mode="wmsr", F=F)
        last_mean = aggregate_hypers(reported, mode="mean", F=F)

        print(f"Reported hypers: {reported}")
        print(f"WMSR hypers    : {last_wmsr}")
        print(f"Mean hypers    : {last_mean}")
        print(f"Used budget so far: {used_budget:.2f} / {total_budget:.2f}")

        for rb in robots:
            if len(rb.X) >= 8:
                gp_fused = build_gp_from_hypers(*last_wmsr)
                X = np.array(rb.X, dtype=float)
                y = np.array(rb.y, dtype=float)
                gp_fused.fit(X, y)
                rb.gp = gp_fused

    print("\n=== FINAL SUMMARY ===")
    print(f"Total budget used: {used_budget:.2f} / {total_budget:.2f}")
    for rb in robots:
        print(f"Robot {rb.rid}: samples={len(rb.X)}, final_node={rb.current_node}")

    good_ids = [i for i in range(n_robots) if i not in faulty_set]

    plot_paper_style_paths_5panel(
        outdir=OUTDIR,
        graph=graph,
        all_stage_paths=all_stage_paths,
        chosen_meeting_nodes=chosen_meeting_nodes,
    )

    plot_gp_surfaces_4panel_paper(
        outdir=OUTDIR,
        graph=graph,
        robots=robots,
        good_ids=good_ids,
        wmsr_hypers=last_wmsr,
        mean_hypers=last_mean,
        init_points=20,
        grid_n=65,
    )

    print(f"\nSaved plots in: {OUTDIR.resolve()}")
    print("Generated files:")
    print("- paper_fig5_paths.png")
    print("- paper_fig6_gp_4panel.png")
    print("- paper_fig6_difference.png")


if __name__ == "__main__":
    main()