# multi_agent_resilience.py
# FULL WORKING CODE (single file)
# - Fixes cKDTree index bug
# - Clean, paper-style path plots (separate + readable replanning)
# - Keeps your 3D 4-panel figure

from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

from matplotlib.ticker import MaxNLocator


# -----------------------------
# GP helpers
# -----------------------------
def build_gp(amp=1.0, ls=1.0, noi=1.0, *, optimize=True, seed=0):
    kernel = (
        C(amp, (1e-3, 1e3))
        * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=noi, noise_level_bounds=(1e-6, 1e2))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=5 if optimize else 0,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        random_state=seed,
    )


def extract_hypers(gp):
    k = gp.kernel_
    amp = float(k.k1.k1.constant_value)
    ls = float(k.k1.k2.length_scale)
    noi = float(k.k2.noise_level)
    return amp, ls, noi


# -----------------------------
# Resilient aggregation
# -----------------------------
def wmsr(vals, F):
    vals = sorted([float(v) for v in vals])
    n = len(vals)
    if n == 0:
        raise ValueError("wmsr got empty list.")
    if 2 * F >= n:
        return float(np.mean(vals))
    return float(np.mean(vals[F : n - F]))


def aggregate(hypers_list, mode="mean", F=1):
    amps = [h[0] for h in hypers_list]
    lss = [h[1] for h in hypers_list]
    nois = [h[2] for h in hypers_list]

    if mode == "mean":
        return float(np.mean(amps)), float(np.mean(lss)), float(np.mean(nois))
    if mode == "wmsr":
        return wmsr(amps, F), wmsr(lss, F), wmsr(nois, F)
    raise ValueError("mode must be 'mean' or 'wmsr'")


# -----------------------------
# Fault model
# -----------------------------
def spatial_attack(x_new, bias):
    x1 = float(x_new[0, 0])
    x2 = float(x_new[0, 1])
    return float(bias * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2))


# -----------------------------
# Plot utilities
# -----------------------------
def compress_path(P, keep_every=3):
    """Keep every k-th point + ensure first/last included."""
    P = np.asarray(P, dtype=float)
    if P.shape[0] <= 2:
        return P
    idx = np.arange(0, P.shape[0], keep_every, dtype=int)
    if idx[-1] != P.shape[0] - 1:
        idx = np.append(idx, P.shape[0] - 1)
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    return P[idx]


def meeting_area_points(meeting_center, *, w=2.0, h=2.0, step=0.5):
    cx, cy = float(meeting_center[0]), float(meeting_center[1])
    xs = np.arange(cx - w / 2, cx + w / 2 + 1e-9, step)
    ys = np.arange(cy - h / 2, cy + h / 2 + 1e-9, step)
    XX, YY = np.meshgrid(xs, ys)
    return np.c_[XX.reshape(-1), YY.reshape(-1)]


def rmse(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -----------------------------
# Multi-agent simulation
# -----------------------------
def run_multi_agent(
    mode,
    X0,
    y0,
    grid,
    y_actual,
    *,
    N=5,
    T=50,                 # exploration steps only
    meeting_every=10,
    F=1,
    n_faulty=1,
    faulty_set=None,
    faulty_bias=7.0,
    faulty_noise_std=3.0,
    clean_noise_std=0.0,
    seed=0,
    freeze_after_meeting=True,
    fuse_predictions=True,
    verbose=True,
    move_radius=2.0,
    meeting_point=np.array([8.0, 0.8], dtype=float),
    meet_steps=6,
    meeting_radius=0.25,
):
    if mode not in ("mean", "wmsr"):
        raise ValueError("mode must be 'mean' or 'wmsr'")

    rng_global = np.random.default_rng(seed)
    tree = cKDTree(grid)

    # choose faulty agents
    if faulty_set is None:
        idxs = np.arange(N)
        rng_global.shuffle(idxs)
        faulty_set = set(int(i) for i in idxs[:n_faulty])
    else:
        faulty_set = set(int(i) for i in faulty_set)

    # IMPORTANT:
    # Start position should be SAME for all robots.
    # We'll use the FIRST initial point and snap it to nearest grid node.
    start_pos = np.array(X0[0], dtype=float).copy()
    _, idx_start = tree.query(start_pos.reshape(1, 2))
    idx_start = int(np.atleast_1d(idx_start)[0])  # FIX: make scalar
    start_pos = grid[idx_start].copy()

    agents = []
    for i in range(N):
        # Each agent begins with SAME initial dataset (as you had)
        X = X0.copy()
        y = y0.copy()

        used = np.zeros(grid.shape[0], dtype=bool)
        _, idx0 = tree.query(X)
        used[np.asarray(idx0, dtype=int)] = True

        gp = build_gp(optimize=True, seed=seed + i)

        agents.append(
            {
                "id": i,
                "faulty": (i in faulty_set),
                "X": X,
                "y": y,
                "used": used,
                "gp": gp,
                "rng": np.random.default_rng(seed + 1000 + i),
                "pos": start_pos.copy(),
                "path": [start_pos.copy()],
            }
        )

    meet_xy = []
    explore_step = 0

    first_meet_path_snapshot = None
    first_meet_index_in_path = None

    while explore_step < T:

        # Meeting trigger based on exploration steps
        if (explore_step > 0) and (explore_step % meeting_every == 0):

            # rendezvous travel (extra steps, does NOT count toward exploration)
            for _ in range(int(meet_steps)):
                for a in agents:
                    vec = meeting_point - a["pos"]
                    dist = float(np.linalg.norm(vec))
                    if dist > 1e-12:
                        step = min(move_radius, dist)
                        new_pos = a["pos"] + (vec / dist) * step
                    else:
                        new_pos = a["pos"]

                    if np.linalg.norm(meeting_point - new_pos) <= meeting_radius:
                        new_pos = meeting_point.copy()

                    # snap to nearest grid point
                    _, idxn = tree.query(new_pos.reshape(1, 2))
                    idxn = int(np.atleast_1d(idxn)[0])  # FIX: scalar
                    new_pos = grid[idxn].copy()

                    a["pos"] = new_pos.copy()
                    a["path"].append(a["pos"].copy())

            meet_xy.append(meeting_point.copy())

            # snapshot pre-meeting (first meeting only)
            if first_meet_path_snapshot is None:
                first_meet_index_in_path = {a["id"]: len(a["path"]) for a in agents}
                first_meet_path_snapshot = {
                    a["id"]: np.array(a["path"], dtype=float).copy() for a in agents
                }

            # fuse hypers
            hypers = []
            for a in agents:
                a["gp"].fit(a["X"], a["y"])
                hypers.append(extract_hypers(a["gp"]))

            agg_amp, agg_ls, agg_noi = aggregate(hypers, mode=mode, F=F)

            if verbose:
                print(
                    f"[{mode}] MEET+FUSE at explore_step={explore_step:02d} | faulty={sorted(list(faulty_set))} | "
                    f"agg=(amp={agg_amp:.3f}, ls={agg_ls:.3f}, noi={agg_noi:.3f})"
                )

            # all agents restart from aggregated hyperparams
            for i, a in enumerate(agents):
                a["gp"] = build_gp(
                    amp=agg_amp,
                    ls=agg_ls,
                    noi=agg_noi,
                    optimize=(not freeze_after_meeting),
                    seed=seed + i,
                )

        # exploration step (counts toward T)
        for a in agents:
            a["gp"].fit(a["X"], a["y"])
            _, y_std = a["gp"].predict(grid, return_std=True)

            y_std_masked = y_std.copy()
            y_std_masked[a["used"]] = -np.inf
            jitter = 1e-9 * a["rng"].standard_normal(size=y_std_masked.shape[0])

            # local move constraint
            pos = a["pos"].reshape(1, 2)
            d = np.linalg.norm(grid - pos, axis=1)
            feasible = (d <= move_radius) & (~a["used"])

            if np.any(feasible):
                y_std_local = y_std.copy()
                y_std_local[~feasible] = -np.inf
                idx_next = int(np.argmax(y_std_local + jitter))
            else:
                idx_next = int(np.argmax(y_std_masked + jitter))

            x_new = grid[idx_next].reshape(1, 2)

            # move + record
            a["pos"] = x_new.reshape(2,)
            a["path"].append(a["pos"].copy())

            # measurement
            y_true = float(y_actual[idx_next])
            if a["faulty"]:
                attack = spatial_attack(x_new, faulty_bias)
                y_meas = y_true + attack + float(a["rng"].normal(0, faulty_noise_std))
            else:
                y_meas = y_true + float(a["rng"].normal(0, clean_noise_std))

            a["X"] = np.vstack([a["X"], x_new])
            a["y"] = np.append(a["y"], y_meas)
            a["used"][idx_next] = True

        explore_step += 1

    # final predictions per agent
    preds = []
    for a in agents:
        a["gp"].fit(a["X"], a["y"])
        preds.append(a["gp"].predict(grid))
    preds = np.vstack(preds)  # (N, M)

    # fuse predictions (pixel-wise) if requested
    if fuse_predictions:
        y_fused = np.zeros(preds.shape[1])
        for j in range(preds.shape[1]):
            y_fused[j] = wmsr(preds[:, j], F) if mode == "wmsr" else float(np.mean(preds[:, j]))
    else:
        y_fused = np.mean(preds, axis=0)

    paths_all = {a["id"]: np.array(a["path"], dtype=float) for a in agents}
    meet_xy = np.array(meet_xy, dtype=float) if len(meet_xy) else np.empty((0, 2), dtype=float)

    # pre/post split relative to first meeting
    if first_meet_path_snapshot is None:
        paths_pre = {rid: paths_all[rid].copy() for rid in paths_all}
        paths_post = {rid: np.empty((0, 2), dtype=float) for rid in paths_all}
    else:
        paths_pre = first_meet_path_snapshot
        paths_post = {}
        for rid, P in paths_all.items():
            cut = int(first_meet_index_in_path[rid])
            # start post segment from meeting position (last point of pre)
            paths_post[rid] = P[cut - 1 :].copy()

    return y_fused, paths_all, meet_xy, paths_pre, paths_post, start_pos.copy()


# -----------------------------
# 3D panels
# -----------------------------
def plot_3d_panel(ax, Z, title, elev=25, azim=-135):
    n = Z.shape[0]
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_title(title, pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


# -----------------------------
# PAPER-STYLE path plotting
# -----------------------------
def _bg_heat(ax, grid, Z_bg):
    xs = np.unique(grid[:, 0])
    ys = np.unique(grid[:, 1])
    ax.pcolormesh(xs, ys, Z_bg, shading="nearest")
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_aspect("equal", adjustable="box")

    # IMPORTANT: do NOT label every tick (that’s what made yours unreadable)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(True, linewidth=0.4, alpha=0.25)


def plot_single_robot_paper(
    *,
    grid,
    Z_bg,
    robot_id,
    path,
    start_pos,
    meeting_center,
    meeting_pts,
    title,
    save_path,
    keep_every=3,
    line_color="black",
):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), dpi=220)
    _bg_heat(ax, grid, Z_bg)

    ax.set_title(title, fontsize=18, pad=8)
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)

    # meeting area (red dotted grid)
    ax.scatter(meeting_pts[:, 0], meeting_pts[:, 1], s=10, c="red", marker=".", zorder=5, label="Meeting Area")

    # meeting location (pink square)
    ax.scatter([meeting_center[0]], [meeting_center[1]], s=120, marker="s",
               facecolor="#f7a6b8", edgecolor="black", linewidth=0.9, zorder=7, label="Meeting Location")

    # start (white circle)
    ax.scatter([start_pos[0]], [start_pos[1]], s=120, marker="o",
               facecolor="white", edgecolor="black", linewidth=1.2, zorder=8, label="Start")

    # robot path (downsample to look like paper)
    P = compress_path(path, keep_every=keep_every)
    ax.plot(P[:, 0], P[:, 1], lw=4.0, color=line_color, zorder=6, solid_capstyle="round", label=f"robot{robot_id}")

    # ONLY show small waypoint squares (not huge clutter)
    ax.scatter(P[:, 0], P[:, 1], s=40, marker="s", facecolor="#f7a6b8", edgecolor="none", zorder=7)

    ax.legend(loc="lower right", fontsize=11, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {save_path}")


def plot_replanning_after_meeting_paper(
    *,
    grid,
    Z_bg,
    paths_post,
    start_pos,
    meeting_center,
    meeting_pts,
    title,
    save_path,
    keep_every=4,
):
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.4), dpi=220)
    _bg_heat(ax, grid, Z_bg)

    ax.set_title(title, fontsize=18, pad=8)
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)

    ax.scatter(meeting_pts[:, 0], meeting_pts[:, 1], s=10, c="red", marker=".", zorder=5, label="Meeting Area")
    ax.scatter([meeting_center[0]], [meeting_center[1]], s=140, marker="s",
               facecolor="#f7a6b8", edgecolor="black", linewidth=0.9, zorder=7, label="Meeting Location")
    ax.scatter([start_pos[0]], [start_pos[1]], s=120, marker="o",
               facecolor="white", edgecolor="black", linewidth=1.2, zorder=8, label="Start")

    colors = ["black", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

    for i, rid in enumerate(sorted(paths_post.keys())):
        P = paths_post[rid]
        if P.size == 0:
            continue
        c = colors[i % len(colors)]
        P2 = compress_path(P, keep_every=keep_every)
        ax.plot(P2[:, 0], P2[:, 1], lw=4.0, color=c, zorder=6, solid_capstyle="round", label=f"robot{rid}")

        # In combined replanning plot: DO NOT add waypoint squares (it causes clutter)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {save_path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    BASE_DIR = Path(__file__).parent
    mat_path = BASE_DIR / "matlab_workspace.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Could not find {mat_path}. Put matlab_workspace.mat in the same folder as this file.")

    mat = scipy.io.loadmat(mat_path)
    g = mat["g"][0, 0]

    X0 = np.array(g["xi"], dtype=float)
    y0 = np.array(g["yi"], dtype=float).reshape(-1)
    grid = np.array(g["grid"], dtype=float)
    y_actual = np.array(g["yactual"], dtype=float).reshape(-1)

    GRID_RES = int(np.sqrt(grid.shape[0]))
    if GRID_RES * GRID_RES != grid.shape[0]:
        raise ValueError("Expected grid to be square (e.g., 25x25).")

    Z_actual = y_actual.reshape(GRID_RES, GRID_RES)

    # initial GP
    gp_init = build_gp(optimize=True, seed=123)
    gp_init.fit(X0, y0)
    y_init = gp_init.predict(grid)
    Z_init = y_init.reshape(GRID_RES, GRID_RES)

    # -----------------------------
    # SETTINGS
    # -----------------------------
    N = 3
    T = 50
    meeting_every = 10

    n_faulty = 1
    F = 1

    faulty_bias = 7.0
    faulty_noise_std = 5.0
    clean_noise_std = 0.0

    seed = 0
    freeze_after_meeting = True
    fuse_predictions = True

    move_radius = 2.0
    meeting_point = np.array([8.0, 0.8], dtype=float)
    meet_steps = 6
    meeting_radius = 0.25

    meet_pts = meeting_area_points(meeting_point, w=2.0, h=2.0, step=0.5)

    # FIX faulty set ONCE so mean vs wmsr is comparable
    rng = np.random.default_rng(seed)
    idxs = np.arange(N)
    rng.shuffle(idxs)
    faulty_set = set(int(i) for i in idxs[:n_faulty])
    print(f"[main] fixed faulty_set={sorted(list(faulty_set))} (N={N}, n_faulty={n_faulty}, F={F})")

    # -----------------------------
    # Non-resilient (mean)
    # -----------------------------
    y_nonres, paths_all_nonres, meet_xy_nonres, paths_pre_nonres, paths_post_nonres, start_pos = run_multi_agent(
        "mean",
        X0, y0, grid, y_actual,
        N=N, T=T, meeting_every=meeting_every,
        F=F, n_faulty=n_faulty, faulty_set=faulty_set,
        faulty_bias=faulty_bias, faulty_noise_std=faulty_noise_std,
        clean_noise_std=clean_noise_std,
        seed=seed,
        freeze_after_meeting=freeze_after_meeting,
        fuse_predictions=fuse_predictions,
        verbose=True,
        move_radius=move_radius,
        meeting_point=meeting_point,
        meet_steps=meet_steps,
        meeting_radius=meeting_radius,
    )
    Z_nonres = y_nonres.reshape(GRID_RES, GRID_RES)

    # -----------------------------
    # Resilient (W-MSR)
    # -----------------------------
    y_res, paths_all_res, meet_xy_res, paths_pre_res, paths_post_res, _ = run_multi_agent(
        "wmsr",
        X0, y0, grid, y_actual,
        N=N, T=T, meeting_every=meeting_every,
        F=F, n_faulty=n_faulty, faulty_set=faulty_set,
        faulty_bias=faulty_bias, faulty_noise_std=faulty_noise_std,
        clean_noise_std=clean_noise_std,
        seed=seed,
        freeze_after_meeting=freeze_after_meeting,
        fuse_predictions=fuse_predictions,
        verbose=True,
        move_radius=move_radius,
        meeting_point=meeting_point,
        meet_steps=meet_steps,
        meeting_radius=meeting_radius,
    )
    Z_res = y_res.reshape(GRID_RES, GRID_RES)

    # -----------------------------
    # Metrics
    # -----------------------------
    print(f"[metric] RMSE init     : {rmse(y_init, y_actual):.4f}")
    print(f"[metric] RMSE nonres   : {rmse(y_nonres, y_actual):.4f}")
    print(f"[metric] RMSE resilient: {rmse(y_res, y_actual):.4f}")

    # -----------------------------
    # Figure: 3D 4-panel
    # -----------------------------
    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_3d_panel(ax1, Z_actual, "(a) Actual environmental GP")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_3d_panel(ax2, Z_init, "(b) Initial knowledge of the GP")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    plot_3d_panel(ax3, Z_nonres, "(c) Learned GP — non-resilient (mean fusion)")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    plot_3d_panel(ax4, Z_res, "(d) Learned GP — resilient (W-MSR fusion)")

    plt.tight_layout()
    out3d = BASE_DIR / "paper_style_3d_4panel.png"
    plt.savefig(out3d, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[main] saved: {out3d}")

    # -----------------------------
    # Paper-style paths (RESILIENT) — each robot separately
    # -----------------------------
    Z_bg = Z_actual  # background heatmap like paper
    paper_colors = ["black", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    keep_every = 3  # smaller = more waypoints (paper clarity). Try 2/3/4.

    for rid in sorted(paths_pre_res.keys()):
        plot_single_robot_paper(
            grid=grid,
            Z_bg=Z_bg,
            robot_id=rid,
            path=paths_pre_res[rid],
            start_pos=start_pos,
            meeting_center=meeting_point,
            meeting_pts=meet_pts,
            title=f"Path for robot {rid}",
            save_path=BASE_DIR / f"path_robot_{rid}_paper.png",
            keep_every=keep_every,
            line_color=paper_colors[rid % len(paper_colors)],
        )

    # -----------------------------
    # Paper-style "Re-planning after meeting"
    # -----------------------------
    plot_replanning_after_meeting_paper(
        grid=grid,
        Z_bg=Z_bg,
        paths_post=paths_post_res,
        start_pos=start_pos,
        meeting_center=meeting_point,
        meeting_pts=meet_pts,
        title="Re-planning after meeting",
        save_path=BASE_DIR / "replanning_after_meeting_paper.png",
        keep_every=4,
    )

    print("[main] done. Check generated PNGs:")
    print("   - paper_style_3d_4panel.png")
    print("   - path_robot_*_paper.png")
    print("   - replanning_after_meeting_paper.png")


if __name__ == "__main__":
    main()
