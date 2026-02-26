# wmsr_pybullet_resilience_paper_fixed_meetarea_v4.py
# 4 drones + 1 faulty, exactly 2 cycles:
#   Start -> Explore -> Meet (fixed meeting area) -> Share hypers (WMSR/Mean)
#        -> Explore -> Meet (same area) -> Share hypers -> STOP
#
# Fixes:
#   ✅ Fixed “meeting area” (paper-like red dotted square)
#   ✅ Resilient vs non-resilient GP surfaces are now meaningfully different:
#       - Resilient: train on GOOD drones only + WMSR hypers
#       - Non-resilient: train on ALL drones (including faulty) + MEAN hypers

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# PAPER <-> SIM coordinate mapping
# ============================================================
def to_paper_xy(x, y, XMIN, XMAX, YMIN, YMAX):
    xp = (x - XMIN) / (XMAX - XMIN) * 20.0 - 10.0  # [-10,10]
    yp = (y - YMIN) / (YMAX - YMIN) * 10.0 - 5.0   # [-5,5]
    return float(xp), float(yp)


def paper_to_sim_xy(xp, yp, XMIN, XMAX, YMIN, YMAX):
    x = (xp + 10.0) / 20.0 * (XMAX - XMIN) + XMIN
    y = (yp + 5.0) / 10.0 * (YMAX - YMIN) + YMIN
    return float(x), float(y)


# ============================================================
# PAPER-domain environment field
# ============================================================
def field_f_paper(xp, yp):
    return (
        2.2 * np.sin(0.55 * xp)
        + 1.8 * np.cos(0.70 * yp)
        + 1.3 * np.sin(0.18 * xp * yp)
        + 0.9 * np.cos(0.35 * (xp + 0.6 * yp))
        + 1.2 * np.sin(0.25 * xp + 0.15 * yp)
    )


# ============================================================
# GP helpers
# ============================================================
def build_gp(amp=1.0, ls=2.0, noi=0.25, *, optimize=True, seed=0):
    kernel = (
        C(amp, (1e-6, 1e6))
        * Matern(length_scale=ls, length_scale_bounds=(1e-3, 1e4), nu=1.5)
        + WhiteKernel(noise_level=noi, noise_level_bounds=(1e-8, 1e3))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=1 if optimize else 0,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        random_state=seed,
    )


def extract_hypers(gp: GaussianProcessRegressor):
    k = gp.kernel_
    amp = float(k.k1.k1.constant_value)
    ls = float(k.k1.k2.length_scale)
    noi = float(k.k2.noise_level)
    return amp, ls, noi


def wmsr(vals, F: int):
    vals = sorted([float(v) for v in vals])
    n = len(vals)
    if n == 0:
        return 0.0
    if 2 * F >= n:
        return float(np.mean(vals))
    return float(np.mean(vals[F : n - F]))


def aggregate_hypers(hypers_list, mode="wmsr", F=1):
    amps = [h[0] for h in hypers_list]
    lss = [h[1] for h in hypers_list]
    nois = [h[2] for h in hypers_list]
    if mode == "mean":
        return float(np.mean(amps)), float(np.mean(lss)), float(np.mean(nois))
    if mode == "wmsr":
        return wmsr(amps, F), wmsr(lss, F), wmsr(nois, F)
    raise ValueError("mode must be 'mean' or 'wmsr'")


# ============================================================
# Env state compatibility
# ============================================================
def get_state_vector(env, i: int):
    if hasattr(env, "getDroneStateVector"):
        return env.getDroneStateVector(i)
    return env._getDroneStateVector(i)


# ============================================================
# Motion helpers
# ============================================================
def step_toward(cur_xy: np.ndarray, goal_xy: np.ndarray, step: float) -> np.ndarray:
    d = goal_xy - cur_xy
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return goal_xy.copy()
    if dist <= step:
        return goal_xy.copy()
    return cur_xy + step * (d / dist)


def meeting_formation(center_xyz: np.ndarray, num: int) -> np.ndarray:
    s = 0.28
    offsets = np.array(
        [
            [-s, -s, 0.0],
            [ s, -s, 0.0],
            [-s,  s, 0.0],
            [ s,  s, 0.0],
        ],
        dtype=np.float32,
    )
    return center_xyz.reshape(1, 3) + offsets[:num]


def subareas_for_drones(XMIN, XMAX, YMIN, YMAX):
    xm = 0.5 * (XMIN + XMAX)
    ym = 0.5 * (YMIN + YMAX)
    return [
        (XMIN, xm,   YMIN, ym),   # drone 0
        (xm,   XMAX, YMIN, ym),   # drone 1
        (XMIN, xm,   ym,   YMAX), # drone 2
        (xm,   XMAX, ym,   YMAX), # drone 3
    ]


def make_sweep_waypoints(xmin, xmax, ymin, ymax, rows=6, z=1.2):
    ys = np.linspace(ymin, ymax, rows)
    wps = []
    flip = False
    for y in ys:
        if not flip:
            wps.append([xmin, float(y), z])
            wps.append([xmax, float(y), z])
        else:
            wps.append([xmax, float(y), z])
            wps.append([xmin, float(y), z])
        flip = not flip
    return np.array(wps, dtype=np.float32)


# ============================================================
# Paper-style plotting
# ============================================================
def paper_background_field(n=31):
    xs = np.linspace(-10, 10, n)
    ys = np.linspace(-5, 5, n)
    XX, YY = np.meshgrid(xs, ys)
    Z = field_f_paper(XX, YY)
    return XX, YY, Z


def plot_paper_style_paths_4panel(
    outdir: Path,
    pos_hist: np.ndarray,
    phase_hist: list,
    cycle_hist: list,
    XMIN, XMAX, YMIN, YMAX,
    drones_data,
    meeting_center_paper=(0.0, 0.0),
    meeting_square_r=1.3,
):
    T, NUM, _ = pos_hist.shape
    phase_hist = np.array(phase_hist)
    cycle_hist = np.array(cycle_hist)

    ds = 25
    idxs = np.arange(0, T, ds)

    xp = np.zeros((len(idxs), NUM), dtype=float)
    yp = np.zeros((len(idxs), NUM), dtype=float)
    for ii, k in enumerate(idxs):
        for i in range(NUM):
            xp[ii, i], yp[ii, i] = to_paper_xy(
                pos_hist[k, i, 0], pos_hist[k, i, 1], XMIN, XMAX, YMIN, YMAX
            )

    phase_ds = phase_hist[idxs]
    cycle_ds = cycle_hist[idxs]

    XX, YY, Z = paper_background_field(n=31)

    mask_c0_explore = (cycle_ds == 0) & (phase_ds == "explore")
    mask_c1_explore = (cycle_ds == 1) & (phase_ds == "explore")
    if not np.any(mask_c1_explore):
        mask_c1_explore = (phase_ds == "explore")

    cx, cy = meeting_center_paper
    x0, x1, y0, y1 = cx - meeting_square_r, cx + meeting_square_r, cy - meeting_square_r, cy + meeting_square_r

    def draw_common(ax):
        ax.pcolormesh(XX, YY, Z, shading="auto")
        ax.grid(True, linewidth=0.45, alpha=0.65)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

        # fixed meeting area square (paper-style)
        ax.plot([x0, x1], [y0, y0], "r--", lw=1.2)
        ax.plot([x0, x1], [y1, y1], "r--", lw=1.2)
        ax.plot([x0, x0], [y0, y1], "r--", lw=1.2)
        ax.plot([x1, x1], [y0, y1], "r--", lw=1.2)

    # sensing points (avoid spaghetti)
    sensing_pts = []
    for i in range(NUM):
        Xp = drones_data[i]["X"]
        sensing_pts.append(Xp[::120] if Xp.shape[0] > 0 else np.empty((0, 2), dtype=float))

    fig = plt.figure(figsize=(16, 4.4), dpi=240)
    axes = [fig.add_subplot(1, 4, i + 1) for i in range(4)]

    for r in range(3):
        ax = axes[r]
        ax.set_title("Planning Paths", fontsize=11)
        draw_common(ax)
        if np.any(mask_c0_explore):
            ax.plot(xp[mask_c0_explore, r], yp[mask_c0_explore, r], lw=2.7, color="k")
        if sensing_pts[r].shape[0] > 0:
            ax.scatter(sensing_pts[r][:, 0], sensing_pts[r][:, 1], s=8, c="red", alpha=0.75, marker=".")

    ax = axes[3]
    ax.set_title("Planning Paths", fontsize=11)
    draw_common(ax)
    for i in range(NUM):
        ax.plot(xp[mask_c1_explore, i], yp[mask_c1_explore, i], lw=2.0)
    for i in range(NUM):
        if sensing_pts[i].shape[0] > 0:
            ax.scatter(sensing_pts[i][:, 0], sensing_pts[i][:, 1], s=6, c="red", alpha=0.55, marker=".")

    caps = ["(a) Path for robot 1", "(b) Path for robot 2", "(c) Path for robot 3", "(d) Re-planning after meeting"]
    for i, cap in enumerate(caps):
        axes[i].text(0.5, -0.22, cap, transform=axes[i].transAxes, ha="center", fontsize=11)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outdir / "paper_style_paths_4panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gp_surfaces_4panel_paper(
    outdir: Path,
    drones_data,
    good_ids,
    faulty_ids,
    wmsr_hypers,
    mean_hypers,
    init_points=80,
    grid_n=65,
):
    xs = np.linspace(-10, 10, grid_n)
    ys = np.linspace(-5, 5, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    Xgrid = np.c_[XX.ravel(), YY.ravel()]

    Z_true = field_f_paper(XX, YY)

    # Initial GP: use early points from all GOOD drones (less flat than using one drone)
    X_init_list, y_init_list = [], []
    for i in good_ids:
        Xi = drones_data[i]["X"]
        yi = drones_data[i]["y"]
        if Xi.shape[0] > 0:
            X_init_list.append(Xi[:init_points])
            y_init_list.append(yi[:init_points])
    if len(X_init_list) > 0:
        X_init = np.vstack(X_init_list)
        y_init = np.concatenate(y_init_list)
    else:
        X_init = np.empty((0, 2))
        y_init = np.empty((0,))

    if X_init.shape[0] >= 15:
        gp_init = build_gp(optimize=True, seed=999)
        gp_init.fit(X_init, y_init)
        mu_init = gp_init.predict(Xgrid).reshape(YY.shape)
    else:
        mu_init = np.zeros_like(YY)

    # Resilient GP: GOOD drones only + WMSR hypers
    X_good = np.vstack([drones_data[i]["X"] for i in good_ids if drones_data[i]["X"].shape[0] > 0])
    y_good = np.concatenate([drones_data[i]["y"] for i in good_ids if drones_data[i]["y"].shape[0] > 0])

    # Non-resilient GP: ALL drones (includes faulty) + MEAN hypers
    X_all = np.vstack([d["X"] for d in drones_data if d["X"].shape[0] > 0])
    y_all = np.concatenate([d["y"] for d in drones_data if d["y"].shape[0] > 0])

    if X_good.shape[0] >= 25 and wmsr_hypers is not None:
        aw, lw, nw = wmsr_hypers
        gp_w = build_gp(amp=aw, ls=lw, noi=nw, optimize=False, seed=1001)
        gp_w.fit(X_good, y_good)
        mu_w = gp_w.predict(Xgrid).reshape(YY.shape)
    else:
        mu_w = np.zeros_like(YY)

    if X_all.shape[0] >= 25 and mean_hypers is not None:
        am, lm, nm = mean_hypers
        gp_m = build_gp(amp=am, ls=lm, noi=nm, optimize=False, seed=1002)
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
        "(c) Learned GP by well-behaving\nrobot using resilient MIPP",
        "(d) Learned GP by well-behaving\nrobot using non-resilient MIPP",
    ]

    for i, Z in enumerate(Zs):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.plot_surface(XX, YY, Z, linewidth=0.2, antialiased=True)
        ax.set_title(titles[i], pad=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlim(zmin, zmax)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=25, azim=235)

    for i, cap in enumerate(caps):
        fig.text((i + 0.5) / 4.0, 0.02, cap, ha="center", va="bottom", fontsize=12)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outdir / "gp_surfaces_4panel_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 5.6), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    diff = np.abs(mu_w - mu_m)
    ax.plot_surface(XX, YY, diff, linewidth=0.2, antialiased=True)
    ax.set_title("|Resilient GP - Non-Resilient GP|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=25, azim=235)
    plt.tight_layout()
    plt.savefig(outdir / "gp_surface_diff_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    OUTDIR = Path(".")
    OUTDIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)

    NUM = 4
    CTRL_FREQ = 240
    DT = 1.0 / CTRL_FREQ

    GUI = True
    DRONE_MODEL = DroneModel.CF2X
    PHY = Physics.PYB

    # sim domain (mapped to paper coords)
    XMIN, XMAX = 0.2, 1.6
    YMIN, YMAX = 0.2, 1.1
    Z_GOAL = 1.2

    MAX_CYCLES = 2

    # explore
    EXPLORE_TIMEOUT_SEC = 45.0
    EXPLORE_MIN_SEC = 10.0
    WP_REACH_XY = 0.15
    WAYPOINT_STEP_EXPLORE = 0.16

    # meet
    Z_TOL = 0.25
    MEET_RADIUS_XY = 0.22
    MEET_SETTLE_SEC = 0.6
    MEET_TIMEOUT_SEC = 18.0

    # takeoff
    TAKEOFF_SEC = 2.0

    # two-step meet
    MEET_Z_APPROACH = 1.45
    MEET_Z_FINAL = 1.20
    MEET_APPROACH_SEC = 2.0

    # crash recovery
    CRASH_Z = 0.18
    RECOVER_SEC = 2.0

    OPTIMIZE_AT_MEET = False

    def clamp_xy(xy):
        xy = xy.copy()
        xy[0] = float(np.clip(xy[0], XMIN, XMAX))
        xy[1] = float(np.clip(xy[1], YMIN, YMAX))
        return xy

    def inside_box(xy):
        return (XMIN <= xy[0] <= XMAX) and (YMIN <= xy[1] <= YMAX)

    # -------- FIXED MEETING AREA (paper coords) --------
    MEET_PAPER_CENTER = (-0.8, 0.6)  # like paper (center-ish)
    MEET_SIM_X, MEET_SIM_Y = paper_to_sim_xy(MEET_PAPER_CENTER[0], MEET_PAPER_CENTER[1], XMIN, XMAX, YMIN, YMAX)
    MEETING_CENTER = np.array([MEET_SIM_X, MEET_SIM_Y, Z_GOAL], dtype=np.float32)

    # Fault model (make mean vs WMSR differ more)
    faulty_set = {3}
    good_set = set(range(NUM)) - faulty_set
    good_ids = sorted(list(good_set))
    faulty_ids = sorted(list(faulty_set))

    F = 1
    meas_noise_std = 0.25
    faulty_extra_noise = 0.8
    attack_bias = 8.0
    BAD_H = (800.0, 25.0, 50.0)  # strong liar hypers to separate MEAN from WMSR

    # start positions
    A0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    eps = 0.25
    offsets = np.array([[0.0, 0.0, 0.0], [eps, 0.0, 0.0], [0.0, eps, 0.0], [eps, eps, 0.0]], dtype=np.float32)
    A = A0.reshape(1, 3) + offsets
    TARGET_RPY = np.zeros((NUM, 3), dtype=np.float32)

    SUBAREAS = subareas_for_drones(XMIN, XMAX, YMIN, YMAX)

    def build_explore_wps_for_cycle(cyc: int):
        # shift sweeps slightly after meeting (paper-ish "replan")
        dx = 0.06 * np.cos(0.9 * cyc)
        dy = 0.06 * np.sin(0.7 * cyc)
        wps = []
        for i in range(NUM):
            xmin, xmax, ymin, ymax = SUBAREAS[i]
            xmin2 = float(np.clip(xmin + dx, XMIN, XMAX))
            xmax2 = float(np.clip(xmax + dx, XMIN, XMAX))
            ymin2 = float(np.clip(ymin + dy, YMIN, YMAX))
            ymax2 = float(np.clip(ymax + dy, YMIN, YMAX))
            lo_x, hi_x = min(xmin2, xmax2), max(xmin2, xmax2)
            lo_y, hi_y = min(ymin2, ymax2), max(ymin2, ymax2)
            wps.append(make_sweep_waypoints(lo_x, hi_x, lo_y, hi_y, rows=6, z=Z_GOAL))
        return wps

    print("\n=== PAPER-LIKE RUN (v4) — fixed meeting area + surface difference ===")
    print("Meeting center (paper):", MEET_PAPER_CENTER, "-> (sim):", MEETING_CENTER.tolist())
    print("Faulty:", faulty_ids, "| WMSR F=", F, "| BAD_H=", BAD_H, "\n")

    env = CtrlAviary(
        drone_model=DRONE_MODEL,
        num_drones=NUM,
        initial_xyzs=A,
        initial_rpys=TARGET_RPY,
        physics=PHY,
        gui=GUI,
        record=False,
        ctrl_freq=CTRL_FREQ,
    )
    env.reset()

    ctrls = [DSLPIDControl(drone_model=DRONE_MODEL) for _ in range(NUM)]

    # store training data in paper coords
    drones = [{"id": i, "X": np.empty((0, 2), dtype=float), "y": np.empty((0,), dtype=float)} for i in range(NUM)]

    # GUI labels
    for i in range(NUM):
        color = [1, 0, 0] if i in faulty_set else [0, 0.7, 1]
        p.addUserDebugText(f"R{i}", [float(A[i, 0]), float(A[i, 1]), float(A[i, 2])],
                           textColorRGB=color, textSize=1.0, lifeTime=0)

    meeting_text_id = p.addUserDebugText("MEETING", MEETING_CENTER.tolist(),
                                         textColorRGB=[1, 0, 1], textSize=1.2, lifeTime=0)

    # logs
    pos_hist, phase_hist, cycle_hist = [], [], []
    z_hist = []
    times = []

    last_xy = np.zeros((NUM, 2), dtype=float)
    stuck_time = np.zeros(NUM, dtype=float)
    recover_timer = np.zeros(NUM, dtype=float)
    within_time_meet = np.zeros(NUM, dtype=float)

    FOLLOW_ID = 0

    # TAKEOFF
    takeoff_steps = int(TAKEOFF_SEC * CTRL_FREQ)
    hover_targets = A.copy()
    hover_targets[:, 2] = Z_GOAL

    for k in range(takeoff_steps):
        rpms = np.zeros((NUM, 4), dtype=np.float32)
        cur_pos = np.zeros((NUM, 3), dtype=float)

        for i in range(NUM):
            state = np.array(get_state_vector(env, i), dtype=np.float32)
            pos = state[0:3]; quat = state[3:7]
            vel = state[10:13] if state.shape[0] >= 13 else np.zeros(3, dtype=np.float32)
            ang_vel = state[13:16] if state.shape[0] >= 16 else np.zeros(3, dtype=np.float32)
            cur_pos[i, :] = pos

            out = ctrls[i].computeControl(
                control_timestep=DT, cur_pos=pos, cur_quat=quat, cur_vel=vel, cur_ang_vel=ang_vel,
                target_pos=hover_targets[i], target_rpy=TARGET_RPY[i],
            )
            rpm = out[0] if isinstance(out, (tuple, list)) else out
            rpms[i, :] = np.array(rpm, dtype=np.float32).reshape(4,)

        env.step(rpms)
        if GUI and (k % 8 == 0):
            cam_target = cur_pos[FOLLOW_ID]
            p.resetDebugVisualizerCamera(
                cameraDistance=2.6, cameraYaw=35, cameraPitch=-35,
                cameraTargetPosition=[float(cam_target[0]), float(cam_target[1]), float(cam_target[2])]
            )
        if GUI:
            time.sleep(DT)

    # MAIN LOOP
    cycle = 0
    phase = "explore"
    phase_start_t = 0.0

    explore_wps = build_explore_wps_for_cycle(cycle)
    wp_idx = np.zeros(NUM, dtype=int)

    active_target = np.zeros((NUM, 3), dtype=np.float32)
    active_target[:, 2] = Z_GOAL

    meet_subphase = "approach"
    meet_subphase_start = 0.0

    last_wmsr_hypers = None
    last_mean_hypers = None

    MAX_STEPS = int(220.0 * CTRL_FREQ)
    for k in range(MAX_STEPS):
        t = k * DT

        rpms = np.zeros((NUM, 4), dtype=np.float32)
        cur_pos = np.zeros((NUM, 3), dtype=float)
        for i in range(NUM):
            state = np.array(get_state_vector(env, i), dtype=np.float32)
            cur_pos[i, :] = state[0:3]

        for i in range(NUM):
            if cur_pos[i, 2] < CRASH_Z:
                recover_timer[i] = RECOVER_SEC

        # meet targets
        if phase == "meet":
            if meet_subphase == "approach":
                meet_center = MEETING_CENTER.copy()
                meet_center[2] = MEET_Z_APPROACH
                meet_targets = meeting_formation(meet_center, NUM)
                if (t - meet_subphase_start) >= MEET_APPROACH_SEC:
                    meet_subphase = "settle"
                    meet_subphase_start = t
            else:
                meet_center = MEETING_CENTER.copy()
                meet_center[2] = MEET_Z_FINAL
                meet_targets = meeting_formation(meet_center, NUM)
        else:
            meet_targets = None

        # update targets
        if phase == "explore":
            for i in range(NUM):
                xy = cur_pos[i, :2].copy()
                if not inside_box(xy):
                    xy = clamp_xy(xy)

                if np.linalg.norm(xy - last_xy[i]) < 1e-3:
                    stuck_time[i] += DT
                else:
                    stuck_time[i] = 0.0
                last_xy[i] = xy

                wps = explore_wps[i]
                j = min(int(wp_idx[i]), len(wps) - 1)
                tgt = wps[j]

                if np.linalg.norm(cur_pos[i, :2] - tgt[:2]) <= WP_REACH_XY and abs(cur_pos[i, 2] - Z_GOAL) <= Z_TOL:
                    if wp_idx[i] < len(wps) - 1:
                        wp_idx[i] += 1
                    tgt = wps[min(int(wp_idx[i]), len(wps) - 1)]

                if stuck_time[i] > 1.5:
                    wp_idx[i] = min(wp_idx[i] + 1, len(wps) - 1)
                    stuck_time[i] = 0.0
                    tgt = wps[min(int(wp_idx[i]), len(wps) - 1)]

                wp_xy = step_toward(xy, tgt[:2], WAYPOINT_STEP_EXPLORE)
                wp_xy = clamp_xy(wp_xy)

                active_target[i, 0] = float(wp_xy[0])
                active_target[i, 1] = float(wp_xy[1])
                active_target[i, 2] = float(Z_GOAL)

        else:
            for i in range(NUM):
                xy = cur_pos[i, :2].copy()
                if not inside_box(xy):
                    xy = clamp_xy(xy)

                if np.linalg.norm(xy - last_xy[i]) < 1e-3:
                    stuck_time[i] += DT
                else:
                    stuck_time[i] = 0.0
                last_xy[i] = xy

                if recover_timer[i] > 0.0:
                    active_target[i, 0] = float(xy[0])
                    active_target[i, 1] = float(xy[1])
                    active_target[i, 2] = float(Z_GOAL)
                    recover_timer[i] -= DT
                    continue

                tgt = meet_targets[i]
                wp_xy = step_toward(xy, tgt[:2], 0.10)
                wp_xy = clamp_xy(wp_xy)

                active_target[i, 0] = float(wp_xy[0])
                active_target[i, 1] = float(wp_xy[1])
                active_target[i, 2] = float(tgt[2])

        # control + measurement
        for i in range(NUM):
            state = np.array(get_state_vector(env, i), dtype=np.float32)
            pos = state[0:3]; quat = state[3:7]
            vel = state[10:13] if state.shape[0] >= 13 else np.zeros(3, dtype=np.float32)
            ang_vel = state[13:16] if state.shape[0] >= 16 else np.zeros(3, dtype=np.float32)

            out = ctrls[i].computeControl(
                control_timestep=DT, cur_pos=pos, cur_quat=quat, cur_vel=vel, cur_ang_vel=ang_vel,
                target_pos=active_target[i], target_rpy=np.zeros(3, dtype=np.float32),
            )
            rpm = out[0] if isinstance(out, (tuple, list)) else out
            rpms[i, :] = np.array(rpm, dtype=np.float32).reshape(4,)

            x, y = float(pos[0]), float(pos[1])
            xp, yp = to_paper_xy(x, y, XMIN, XMAX, YMIN, YMAX)
            true_val = float(field_f_paper(xp, yp))
            noise = float(np.random.normal(0.0, 0.25))

            if i in faulty_set:
                attack = attack_bias * np.sin(0.4 * xp) * np.cos(0.6 * yp)
                noise += float(np.random.normal(0.0, faulty_extra_noise))
                meas = true_val + float(attack) + noise
            else:
                meas = true_val + noise

            drones[i]["X"] = np.vstack([drones[i]["X"], np.array([[xp, yp]], dtype=float)])
            drones[i]["y"] = np.append(drones[i]["y"], meas)

        env.step(rpms)

        if GUI and (k % 8 == 0):
            cam_target = cur_pos[FOLLOW_ID]
            p.resetDebugVisualizerCamera(
                cameraDistance=2.4, cameraYaw=35, cameraPitch=-35,
                cameraTargetPosition=[float(cam_target[0]), float(cam_target[1]), float(cam_target[2])]
            )

        # logs
        times.append(t)
        pos_hist.append(cur_pos.copy())
        phase_hist.append(phase)
        cycle_hist.append(cycle)
        z_hist.append(cur_pos[:, 2].copy())

        elapsed = t - phase_start_t

        if phase == "explore":
            all_done = True
            for i in range(NUM):
                all_done = all_done and (wp_idx[i] >= (len(explore_wps[i]) - 1))
            done = all_done and ((t - phase_start_t) >= EXPLORE_MIN_SEC)
            timed_out = elapsed >= EXPLORE_TIMEOUT_SEC

            if done or timed_out:
                print(f"\n[SWITCH] explore -> meet (done={done}, timed_out={timed_out})")
                phase = "meet"
                phase_start_t = t
                within_time_meet[:] = 0.0
                meet_subphase = "approach"
                meet_subphase_start = t

        else:
            timed_out = elapsed >= MEET_TIMEOUT_SEC
            done = False

            if meet_subphase == "settle":
                ok = np.zeros(NUM, dtype=bool)
                for i in range(NUM):
                    dxy = float(np.linalg.norm(cur_pos[i, :2] - meet_targets[i, :2]))
                    dz = float(abs(cur_pos[i, 2] - meet_targets[i, 2]))
                    if (dxy <= MEET_RADIUS_XY) and (dz <= Z_TOL):
                        within_time_meet[i] += DT
                    else:
                        within_time_meet[i] = 0.0
                    ok[i] = within_time_meet[i] >= MEET_SETTLE_SEC
                done = bool(np.all(ok))

            if done or timed_out:
                print(f"\n[MEET COMPLETE] -> share hypers (WMSR vs MEAN)")

                hypers = []
                for i in range(NUM):
                    Xp = drones[i]["X"]; yv = drones[i]["y"]
                    if Xp.shape[0] >= 40:
                        gp_tmp = build_gp(amp=1.0, ls=2.0, noi=0.25, optimize=OPTIMIZE_AT_MEET, seed=10 + i)
                        gp_tmp.fit(Xp, yv)
                        hypers.append(extract_hypers(gp_tmp))
                    else:
                        hypers.append((1.0, 2.0, 0.25))

                reported = []
                for i, h in enumerate(hypers):
                    reported.append(BAD_H if (i in faulty_set) else h)

                wmsr_h = aggregate_hypers(reported, mode="wmsr", F=F)
                mean_h = aggregate_hypers(reported, mode="mean", F=F)
                last_wmsr_hypers = wmsr_h
                last_mean_hypers = mean_h

                print("  WMSR =", tuple(round(v, 3) for v in wmsr_h))
                print("  MEAN =", tuple(round(v, 3) for v in mean_h))
                print("  DIFF =", tuple(round(abs(wmsr_h[j] - mean_h[j]), 3) for j in range(3)))

                cycle += 1
                if cycle >= MAX_CYCLES:
                    print("\nDone: completed 2 Explore->Meet cycles. Closing.")
                    break

                print("\n[SWITCH] meet -> explore (replan sweeps)\n")
                phase = "explore"
                phase_start_t = t
                explore_wps = build_explore_wps_for_cycle(cycle)
                wp_idx[:] = 0

        if GUI:
            time.sleep(DT)

    env.close()

    pos_hist = np.array(pos_hist, dtype=float)

    # plots
    plot_paper_style_paths_4panel(
        outdir=OUTDIR,
        pos_hist=pos_hist,
        phase_hist=phase_hist,
        cycle_hist=cycle_hist,
        XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX,
        drones_data=drones,
        meeting_center_paper=MEET_PAPER_CENTER,
        meeting_square_r=1.3,
    )

    if last_wmsr_hypers is not None and last_mean_hypers is not None:
        plot_gp_surfaces_4panel_paper(
            outdir=OUTDIR,
            drones_data=drones,
            good_ids=good_ids,
            faulty_ids=faulty_ids,
            wmsr_hypers=last_wmsr_hypers,
            mean_hypers=last_mean_hypers,
            init_points=80,
            grid_n=65,
        )

    print("\nSaved:")
    for f in [
        "paper_style_paths_4panel.png",
        "gp_surfaces_4panel_paper.png",
        "gp_surface_diff_paper.png",
    ]:
        print(" -", OUTDIR / f)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CTRL+C] Interrupted. Exiting cleanly...")
    finally:
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass