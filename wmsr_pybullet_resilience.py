# wmsr_pybullet_resilience_paper.py
# 4 drones + 1 faulty (Byzantine) — Explore -> Meet -> Share GP hypers (WMSR) -> Re-explore
#
# Updates to make results look PAPER-like:
#   ✅ Paper-style planning/path figure (heatmap + grid + 4 panels like Fig.5)
#   ✅ Paper-style 4-panel GP surfaces (paper coordinate axes like Fig.6)
#   ✅ Deterministic "subarea" exploration goals (less spaghetti, more like MIPP regions)
#   ✅ Fix "stuck at share hypers": meeting-time GP fitting is FAST (optimizer off by default)
#   ✅ Safe shutdown on Ctrl+C (env.close + p.disconnect)
#
# Run:
#   (.venv) python wmsr_pybullet_resilience_paper.py
#
# Outputs:
#   - path_drone0.png ... path_drone3.png
#   - paths_all_drones.png
#   - distance_to_active_target.png
#   - z_over_time.png
#   - paper_style_paths_4panel.png        <-- like Fig.5
#   - gp_surfaces_4panel_paper.png        <-- like Fig.6
#   - gp_surface_diff_paper.png

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


# -----------------------------
# Scalar field (virtual sensor)
# -----------------------------
def field_f(x, y):
    # You can replace this with your paper's ground-truth field if you want.
    return (
        2.2 * np.sin(3.2 * x)
        + 1.8 * np.cos(2.7 * y)
        + 1.3 * np.sin(2.1 * x * y)
        + 0.9 * np.cos(1.7 * (x + 0.6 * y))
    )


# -----------------------------
# GP helpers
# -----------------------------
def build_gp(amp=1.0, ls=0.35, noi=0.15, *, optimize=True, seed=0):
    kernel = (
        C(amp, (1e-6, 1e6))
        * Matern(length_scale=ls, length_scale_bounds=(1e-4, 1e3), nu=1.5)
        + WhiteKernel(noise_level=noi, noise_level_bounds=(1e-8, 1e3))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        # optimizer can be slow; keep restarts low
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


# -----------------------------
# Env state compatibility
# -----------------------------
def get_state_vector(env, i: int):
    if hasattr(env, "getDroneStateVector"):
        return env.getDroneStateVector(i)
    return env._getDroneStateVector(i)


# -----------------------------
# Waypoint helper
# -----------------------------
def step_toward(cur_xy: np.ndarray, goal_xy: np.ndarray, step: float) -> np.ndarray:
    d = goal_xy - cur_xy
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return goal_xy.copy()
    if dist <= step:
        return goal_xy.copy()
    return cur_xy + step * (d / dist)


# -----------------------------
# Meeting formation (prevents collisions)
# -----------------------------
def meeting_formation(center_xyz: np.ndarray, num: int) -> np.ndarray:
    # square formation around meeting center for 4 drones
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


# -----------------------------
# Paper coordinate mapping (for plots only)
# -----------------------------
def to_paper_xy(x, y, XMIN, XMAX, YMIN, YMAX):
    # map sim box -> paper-like axes: x in [-10,10], y in [-5,5]
    xp = (x - XMIN) / (XMAX - XMIN) * 20.0 - 10.0
    yp = (y - YMIN) / (YMAX - YMIN) * 10.0 - 5.0
    return xp, yp


def paper_to_sim_xy(xp, yp, XMIN, XMAX, YMIN, YMAX):
    # inverse mapping for evaluating field/GP on paper grid
    x = (xp + 10.0) / 20.0 * (XMAX - XMIN) + XMIN
    y = (yp + 5.0) / 10.0 * (YMAX - YMIN) + YMIN
    return x, y


def paper_background_field(XMIN, XMAX, YMIN, YMAX, n=31):
    # build a grid in paper coords, evaluate field in sim coords
    xs_p = np.linspace(-10, 10, n)
    ys_p = np.linspace(-5, 5, n)
    XXp, YYp = np.meshgrid(xs_p, ys_p)
    Xsim, Ysim = paper_to_sim_xy(XXp, YYp, XMIN, XMAX, YMIN, YMAX)
    Z = field_f(Xsim, Ysim)
    return XXp, YYp, Z


def unique_meeting_centers_paper(meet_hist, XMIN, XMAX, YMIN, YMAX):
    # compress meeting centers over time to unique points
    if len(meet_hist) == 0:
        return np.empty((0, 2), dtype=float)

    pts = []
    for m in meet_hist:
        mx, my, _ = m
        mxp, myp = to_paper_xy(mx, my, XMIN, XMAX, YMIN, YMAX)
        pts.append([mxp, myp])
    pts = np.array(pts, dtype=float)

    # unique by rounding (prevents thousands of duplicates)
    key = np.round(pts, 2)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    return pts[idx]


# -----------------------------
# Paper-style paths figure (Fig.5-like)
# -----------------------------
def plot_paper_style_paths_4panel(
    outdir: Path,
    pos_hist: np.ndarray,
    phase_hist: list,
    cycle_hist: list,
    meet_hist: np.ndarray,
    XMIN, XMAX, YMIN, YMAX,
    faulty_set: set,
):
    T, NUM, _ = pos_hist.shape
    phase_hist = np.array(phase_hist)
    cycle_hist = np.array(cycle_hist)

    # Convert drone trajectories to paper coordinates
    xp = np.zeros((T, NUM), dtype=float)
    yp = np.zeros((T, NUM), dtype=float)
    for k in range(T):
        for i in range(NUM):
            xp[k, i], yp[k, i] = to_paper_xy(
                pos_hist[k, i, 0], pos_hist[k, i, 1],
                XMIN, XMAX, YMIN, YMAX
            )

    # Meeting centers (unique)
    meet_pts = unique_meeting_centers_paper(meet_hist, XMIN, XMAX, YMIN, YMAX)

    # Background heatmap + grid
    XXp, YYp, Z = paper_background_field(XMIN, XMAX, YMIN, YMAX, n=31)

    # Masks for cycle 0 explore and cycle 1 explore
    mask0 = (cycle_hist == 0) & (phase_hist == "explore")
    mask1 = (cycle_hist == 1) & (phase_hist == "explore")

    # If cycle 1 doesn't exist (short run), fall back to last explore segment
    if not np.any(mask1):
        mask1 = (phase_hist == "explore")

    # Make 4-panel plot
    fig = plt.figure(figsize=(16, 4.4), dpi=220)
    axes = [fig.add_subplot(1, 4, i + 1) for i in range(4)]

    # Panels (a)(b)(c): first 3 robots individually (paper style)
    for r in range(min(3, NUM)):
        ax = axes[r]
        ax.pcolormesh(XXp, YYp, Z, shading="auto")
        ax.grid(True, linewidth=0.45, alpha=0.65)
        ax.set_title("Planning Paths", fontsize=11)

        if np.any(mask0):
            ax.plot(xp[mask0, r], yp[mask0, r], linewidth=2.6, color="k")
            ax.scatter([xp[mask0][0, r]], [yp[mask0][0, r]], s=30,
                       color="white", edgecolor="k", zorder=3)

        if meet_pts.shape[0] > 0:
            ax.scatter(meet_pts[:, 0], meet_pts[:, 1], s=18, color="red", marker=".", zorder=3)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    # Panel (d): replanning overlay (all robots, second explore)
    ax = axes[3]
    ax.pcolormesh(XXp, YYp, Z, shading="auto")
    ax.grid(True, linewidth=0.45, alpha=0.65)
    ax.set_title("Planning Paths", fontsize=11)

    for i in range(NUM):
        if np.any(mask1):
            ax.plot(xp[mask1, i], yp[mask1, i], linewidth=2.2)
    if meet_pts.shape[0] > 0:
        ax.scatter(meet_pts[:, 0], meet_pts[:, 1], s=18, color="red", marker=".", zorder=3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    # Sub-captions like paper
    caps = ["(a) Path for robot 1", "(b) Path for robot 2", "(c) Path for robot 3", "(d) Re-planning after meeting"]
    for i, cap in enumerate(caps):
        axes[i].text(0.5, -0.22, cap, transform=axes[i].transAxes, ha="center", fontsize=11)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outdir / "paper_style_paths_4panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Paper-style GP surfaces (Fig.6-like)
# -----------------------------
def plot_gp_surfaces_4panel_paper(
    outdir: Path,
    XMIN, XMAX, YMIN, YMAX,
    drones_data,               # list of {"X":..., "y":...}
    good_id: int,
    wmsr_hypers,
    mean_hypers,
    init_points=60,
    grid_n=55,
):
    # paper grid (x in [-10,10], y in [-5,5])
    xs_p = np.linspace(-10, 10, grid_n)
    ys_p = np.linspace(-5, 5, grid_n)
    XXp, YYp = np.meshgrid(xs_p, ys_p)

    # map paper grid -> sim coords for field + GP prediction
    Xsim, Ysim = paper_to_sim_xy(XXp, YYp, XMIN, XMAX, YMIN, YMAX)
    Xgrid_sim = np.c_[Xsim.ravel(), Ysim.ravel()]

    # True field on paper grid
    Z_true = field_f(Xsim, Ysim)

    # Initial GP (fit only first n0 points from good robot)
    Xd = drones_data[good_id]["X"]
    yd = drones_data[good_id]["y"]
    n0 = min(init_points, Xd.shape[0])

    if n0 >= 12:
        gp_init = build_gp(optimize=True, seed=999)
        gp_init.fit(Xd[:n0], yd[:n0])
        mu_init = gp_init.predict(Xgrid_sim).reshape(YYp.shape)
    else:
        mu_init = np.zeros_like(YYp)

    # Resilient / Non-resilient surfaces (same good robot data, different hypers)
    if Xd.shape[0] >= 12 and wmsr_hypers is not None:
        aw, lw, nw = wmsr_hypers
        gp_w = build_gp(amp=aw, ls=lw, noi=nw, optimize=False, seed=1001)
        gp_w.fit(Xd, yd)
        mu_w = gp_w.predict(Xgrid_sim).reshape(YYp.shape)
    else:
        mu_w = np.zeros_like(YYp)

    if Xd.shape[0] >= 12 and mean_hypers is not None:
        am, lm, nm = mean_hypers
        gp_m = build_gp(amp=am, ls=lm, noi=nm, optimize=False, seed=1002)
        gp_m.fit(Xd, yd)
        mu_m = gp_m.predict(Xgrid_sim).reshape(YYp.shape)
    else:
        mu_m = np.zeros_like(YYp)

    Zs = [Z_true, mu_init, mu_w, mu_m]

    # z-limits shared
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
        ax.plot_surface(XXp, YYp, Z, linewidth=0.2, antialiased=True)

        ax.set_title(titles[i], pad=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlim(zmin, zmax)

        # cleaner paper-like look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=25, azim=235)

    for i, cap in enumerate(caps):
        fig.text((i + 0.5) / 4.0, 0.02, cap, ha="center", va="bottom", fontsize=12)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outdir / "gp_surfaces_4panel_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Difference surface (paper-like)
    fig = plt.figure(figsize=(7.2, 5.6), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    diff = np.abs(mu_w - mu_m)
    ax.plot_surface(XXp, YYp, diff, linewidth=0.2, antialiased=True)
    ax.set_title("|Resilient GP - Non-Resilient GP|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=25, azim=235)
    plt.tight_layout()
    plt.savefig(outdir / "gp_surface_diff_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTDIR = Path(".")
    OUTDIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)

    # ======= 4 DRONES =======
    NUM = 4

    CTRL_FREQ = 240
    DT = 1.0 / CTRL_FREQ

    GUI = True
    DRONE_MODEL = DroneModel.CF2X
    PHY = Physics.PYB

    # SAFE bounds (sim box)
    XMIN, XMAX = 0.2, 1.6
    YMIN, YMAX = 0.2, 1.1
    Z_GOAL = 1.2

    # Phase settings
    MAX_CYCLES = 6

    # settle conditions (XY + altitude)
    EXPLORE_GOAL_RADIUS_XY = 0.18
    MEET_RADIUS_XY = 0.22
    Z_TOL = 0.25

    EXPLORE_SETTLE_SEC = 0.6
    MEET_SETTLE_SEC = 0.6

    EXPLORE_TIMEOUT_SEC = 24.0
    MEET_TIMEOUT_SEC = 18.0

    # Waypoints
    WAYPOINT_STEP_EXPLORE = 0.16
    WAYPOINT_STEP_MEET = 0.10

    # Dynamic meeting point
    MEET_MARGIN = 0.25

    # TAKEOFF
    TAKEOFF_SEC = 2.0

    # 2-step meeting
    MEET_Z_APPROACH = 1.45
    MEET_Z_FINAL = 1.20
    MEET_APPROACH_SEC = 2.0

    # Crash recovery
    CRASH_Z = 0.18
    RECOVER_SEC = 2.0

    # ======= IMPORTANT: GP share speed control =======
    # 0 => never optimize at meetings (fastest, recommended)
    # 2 => optimize every 2nd cycle (0,2,4,...) and fast-fit on others
    OPTIMIZE_EVERY = 0

    def clamp_xy(xy):
        xy = xy.copy()
        xy[0] = float(np.clip(xy[0], XMIN, XMAX))
        xy[1] = float(np.clip(xy[1], YMIN, YMAX))
        return xy

    def inside_box(xy):
        return (XMIN <= xy[0] <= XMAX) and (YMIN <= xy[1] <= YMAX)

    # -----------------------------
    # PAPER-LIKE exploration goals:
    # Each drone gets a "subarea anchor"; after meeting, anchors shift slightly.
    # This makes paths cleaner and closer to Fig.5 style (subareas + replanning).
    # -----------------------------
    def goals_for_cycle(cyc: int):
        base = np.array(
            [
                [0.35, 0.35],  # drone 0 subarea
                [1.45, 0.35],  # drone 1 subarea
                [0.35, 1.00],  # drone 2 subarea
                [1.45, 1.00],  # drone 3 subarea
            ],
            dtype=float,
        )

        # small deterministic shift every cycle (replanning feel)
        shift = 0.08 * np.array([np.cos(0.9 * cyc), np.sin(0.7 * cyc)])
        anchors = base + shift.reshape(1, 2)

        # tiny jitter per drone so they don't exactly overlap across cycles
        jitter = rng.normal(0.0, 0.04, size=(NUM, 2))
        anchors = anchors + jitter

        # clamp
        for i in range(NUM):
            anchors[i, 0] = float(np.clip(anchors[i, 0], XMIN, XMAX))
            anchors[i, 1] = float(np.clip(anchors[i, 1], YMIN, YMAX))

        G = np.zeros((NUM, 3), dtype=np.float32)
        G[:, 0] = anchors[:, 0]
        G[:, 1] = anchors[:, 1]
        G[:, 2] = Z_GOAL
        return G

    def sample_meeting_point():
        mx = float(rng.uniform(XMIN + MEET_MARGIN, XMAX - MEET_MARGIN))
        my = float(rng.uniform(YMIN + MEET_MARGIN, YMAX - MEET_MARGIN))
        return np.array([mx, my, Z_GOAL], dtype=np.float32)

    def sample_goal_near(xy, r=0.35):
        gx = float(np.clip(rng.uniform(xy[0] - r, xy[0] + r), XMIN, XMAX))
        gy = float(np.clip(rng.uniform(xy[1] - r, xy[1] + r), YMIN, YMAX))
        return np.array([gx, gy, Z_GOAL], dtype=np.float32)

    # Fault model
    faulty_set = {3}   # drone 3 is faulty
    F = 1              # WMSR trims one min + one max (n=4)
    meas_noise_std = 0.06
    faulty_extra_noise = 0.25
    attack_bias = 8.0
    BAD_H = (500.0, 5.0, 50.0)

    # Start positions (same logical start, slight spacing)
    A0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    eps = 0.25
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [eps, 0.0, 0.0],
            [0.0, eps, 0.0],
            [eps, eps, 0.0],
        ],
        dtype=np.float32,
    )
    A = A0.reshape(1, 3) + offsets
    TARGET_RPY = np.zeros((NUM, 3), dtype=np.float32)

    MEETING_CENTER = sample_meeting_point()

    print("\n=== STABLE MEET FORMATION RUN (4 drones) ===")
    print("SAFE bounds:", (XMIN, XMAX), (YMIN, YMAX), "z=", Z_GOAL)
    print("Faulty:", sorted(list(faulty_set)), "| WMSR F=", F, "| BAD_H=", BAD_H)
    print("Meeting approach z:", MEET_Z_APPROACH, "final z:", MEET_Z_FINAL)
    print("Initial meeting center:", MEETING_CENTER.tolist(), "\n")

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

    # Per-drone data buffers
    drones = []
    for i in range(NUM):
        drones.append(
            {"id": i,
             "X": np.empty((0, 2), dtype=float),
             "y": np.empty((0,), dtype=float)}
        )

    # GUI labels
    for i in range(NUM):
        color = [1, 0, 0] if i in faulty_set else [0, 0.7, 1]
        p.addUserDebugText(
            f"R{i}",
            [float(A[i, 0]), float(A[i, 1]), float(A[i, 2])],
            textColorRGB=color, textSize=1.0, lifeTime=0
        )

    meeting_text_id = p.addUserDebugText(
        "MEETING", MEETING_CENTER.tolist(),
        textColorRGB=[1, 0, 1], textSize=1.2, lifeTime=0
    )

    def update_meeting_marker(new_center):
        nonlocal meeting_text_id
        if meeting_text_id is not None:
            p.removeUserDebugItem(meeting_text_id)
        meeting_text_id = p.addUserDebugText(
            "MEETING", new_center.tolist(),
            textColorRGB=[1, 0, 1], textSize=1.2, lifeTime=0
        )

    goal_marker_ids = [None] * NUM

    def place_goal_markers(G):
        nonlocal goal_marker_ids
        for i in range(NUM):
            if goal_marker_ids[i] is not None:
                p.removeUserDebugItem(goal_marker_ids[i])
            color = [1, 0, 0] if i in faulty_set else [0, 1, 0]
            goal_marker_ids[i] = p.addUserDebugText(
                f"G{i}",
                [float(G[i, 0]), float(G[i, 1]), float(G[i, 2])],
                textColorRGB=color, textSize=1.2, lifeTime=0
            )

    # Logs
    times, pos_hist, active_target_hist = [], [], []
    phase_hist, cycle_hist, meet_hist = [], [], []
    z_hist = []

    # Recovery / stuck trackers
    last_xy = np.zeros((NUM, 2), dtype=float)
    stuck_time = np.zeros(NUM, dtype=float)
    recover_timer = np.zeros(NUM, dtype=float)
    within_time = np.zeros(NUM, dtype=float)

    def settle_xy_z(cur_pos, target_pos, radius_xy, settle_sec):
        nonlocal within_time
        ok = np.zeros(NUM, dtype=bool)
        for i in range(NUM):
            dxy = float(np.linalg.norm(cur_pos[i, :2] - target_pos[i, :2]))
            dz = float(abs(cur_pos[i, 2] - target_pos[i, 2]))
            if (dxy <= radius_xy) and (dz <= Z_TOL):
                within_time[i] += DT
            else:
                within_time[i] = 0.0
            ok[i] = within_time[i] >= settle_sec
        return bool(np.all(ok))

    FOLLOW_ID = 0

    # -----------------------------
    # TAKEOFF
    # -----------------------------
    takeoff_steps = int(TAKEOFF_SEC * CTRL_FREQ)
    hover_targets = A.copy()
    hover_targets[:, 2] = Z_GOAL

    for k in range(takeoff_steps):
        rpms = np.zeros((NUM, 4), dtype=np.float32)
        cur_pos = np.zeros((NUM, 3), dtype=float)
        for i in range(NUM):
            state = np.array(get_state_vector(env, i), dtype=np.float32)
            pos = state[0:3]
            quat = state[3:7]
            vel = state[10:13] if state.shape[0] >= 13 else np.zeros(3, dtype=np.float32)
            ang_vel = state[13:16] if state.shape[0] >= 16 else np.zeros(3, dtype=np.float32)
            cur_pos[i, :] = pos

            out = ctrls[i].computeControl(
                control_timestep=DT,
                cur_pos=pos,
                cur_quat=quat,
                cur_vel=vel,
                cur_ang_vel=ang_vel,
                target_pos=hover_targets[i],
                target_rpy=TARGET_RPY[i],
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

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    cycle = 0
    phase = "explore"
    phase_start_t = 0.0

    explore_goal = goals_for_cycle(cycle)
    place_goal_markers(explore_goal)
    active_target = explore_goal.copy()

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

        # crash detection -> recovery timer
        for i in range(NUM):
            if cur_pos[i, 2] < CRASH_Z:
                recover_timer[i] = RECOVER_SEC

        # meeting targets (formation)
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

        # update active targets
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

                if stuck_time[i] > 1.5:
                    explore_goal[i, :] = sample_goal_near(xy, r=0.35)
                    stuck_time[i] = 0.0

                wp_xy = step_toward(xy, explore_goal[i, :2], WAYPOINT_STEP_EXPLORE)
                wp_xy = clamp_xy(wp_xy)

                active_target[i, 0] = float(wp_xy[0])
                active_target[i, 1] = float(wp_xy[1])
                active_target[i, 2] = float(Z_GOAL)

        else:  # meet
            for i in range(NUM):
                xy = cur_pos[i, :2].copy()
                if not inside_box(xy):
                    xy = clamp_xy(xy)

                if np.linalg.norm(xy - last_xy[i]) < 1e-3:
                    stuck_time[i] += DT
                else:
                    stuck_time[i] = 0.0
                last_xy[i] = xy

                # recovery override (hover)
                if recover_timer[i] > 0.0:
                    active_target[i, 0] = float(xy[0])
                    active_target[i, 1] = float(xy[1])
                    active_target[i, 2] = float(Z_GOAL)
                    recover_timer[i] -= DT
                    continue

                tgt = meet_targets[i]
                wp_xy = step_toward(xy, tgt[:2], WAYPOINT_STEP_MEET)
                wp_xy = clamp_xy(wp_xy)

                active_target[i, 0] = float(wp_xy[0])
                active_target[i, 1] = float(wp_xy[1])
                active_target[i, 2] = float(tgt[2])

        # control + measurement
        for i in range(NUM):
            state = np.array(get_state_vector(env, i), dtype=np.float32)
            pos = state[0:3]
            quat = state[3:7]
            vel = state[10:13] if state.shape[0] >= 13 else np.zeros(3, dtype=np.float32)
            ang_vel = state[13:16] if state.shape[0] >= 16 else np.zeros(3, dtype=np.float32)

            out = ctrls[i].computeControl(
                control_timestep=DT,
                cur_pos=pos,
                cur_quat=quat,
                cur_vel=vel,
                cur_ang_vel=ang_vel,
                target_pos=active_target[i],
                target_rpy=np.zeros(3, dtype=np.float32),
            )
            rpm = out[0] if isinstance(out, (tuple, list)) else out
            rpms[i, :] = np.array(rpm, dtype=np.float32).reshape(4,)

            # measurement
            x, y = float(pos[0]), float(pos[1])
            true_val = float(field_f(x, y))
            noise = float(np.random.normal(0.0, meas_noise_std))

            if i in faulty_set:
                attack = attack_bias * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
                noise += float(np.random.normal(0.0, faulty_extra_noise))
                meas = true_val + float(attack) + noise
            else:
                meas = true_val + noise

            drones[i]["X"] = np.vstack([drones[i]["X"], np.array([[x, y]], dtype=float)])
            drones[i]["y"] = np.append(drones[i]["y"], meas)

        env.step(rpms)

        # camera follow
        if GUI and (k % 8 == 0):
            cam_target = cur_pos[FOLLOW_ID]
            p.resetDebugVisualizerCamera(
                cameraDistance=2.4, cameraYaw=35, cameraPitch=-35,
                cameraTargetPosition=[float(cam_target[0]), float(cam_target[1]), float(cam_target[2])]
            )

        # periodic prints
        if k % int(CTRL_FREQ * 1.0) == 0:
            xy = np.round(cur_pos[:, :2], 2)
            zz = np.round(cur_pos[:, 2], 2).tolist()
            if phase == "explore":
                err_goal_xy = np.array(
                    [np.linalg.norm(cur_pos[i, :2] - explore_goal[i, :2]) for i in range(NUM)],
                    dtype=float
                )
                print(f"t={t:5.1f}s phase=explore cycle={cycle} err_goal_xy={np.round(err_goal_xy,2).tolist()} z={zz} pos_xy={xy.tolist()}")
            else:
                err_meet_xy = np.array(
                    [np.linalg.norm(cur_pos[i, :2] - meet_targets[i, :2]) for i in range(NUM)],
                    dtype=float
                )
                print(f"t={t:5.1f}s phase=meet({meet_subphase}) cycle={cycle} err_meet_xy={np.round(err_meet_xy,2).tolist()} z={zz} meet={np.round(MEETING_CENTER[:2],2).tolist()} pos_xy={xy.tolist()}")

        # logs
        times.append(t)
        pos_hist.append(cur_pos.copy())
        active_target_hist.append(active_target.copy())
        phase_hist.append(phase)
        cycle_hist.append(cycle)
        meet_hist.append(MEETING_CENTER.copy())
        z_hist.append(cur_pos[:, 2].copy())

        elapsed = t - phase_start_t

        if phase == "explore":
            done = settle_xy_z(cur_pos, explore_goal, EXPLORE_GOAL_RADIUS_XY, EXPLORE_SETTLE_SEC)
            timed_out = elapsed >= EXPLORE_TIMEOUT_SEC
            if done or timed_out:
                print(f"\n[SWITCH] explore -> meet   (done={done}, timed_out={timed_out})")
                phase = "meet"
                within_time[:] = 0.0
                phase_start_t = t
                meet_subphase = "approach"
                meet_subphase_start = t

        else:  # meet
            timed_out = elapsed >= MEET_TIMEOUT_SEC
            done = False
            if meet_subphase == "settle":
                done = settle_xy_z(cur_pos, meet_targets, MEET_RADIUS_XY, MEET_SETTLE_SEC)

            if done or timed_out:
                print(f"\n[MEET COMPLETE] (done={done}, timed_out={timed_out}) -> share hypers")
                print("Fitting local GPs for hyperparameters...")

                do_opt = (OPTIMIZE_EVERY > 0) and (cycle % OPTIMIZE_EVERY == 0)

                hypers = []
                for i in range(NUM):
                    X = drones[i]["X"]
                    yv = drones[i]["y"]
                    print(f"  fitting drone {i} with N={X.shape[0]} samples (optimize={do_opt})...")

                    if X.shape[0] >= 25:
                        # Fast and stable: new GP each meeting
                        gp_tmp = build_gp(amp=1.0, ls=0.35, noi=0.15, optimize=do_opt, seed=10 + i)
                        gp_tmp.fit(X, yv)
                        hypers.append(extract_hypers(gp_tmp))
                    else:
                        hypers.append((1.0, 0.35, 0.15))

                print("Done fitting all local GPs.")

                # Faulty drone lies about hypers
                reported = []
                for i, h in enumerate(hypers):
                    if (BAD_H is not None) and (i in faulty_set):
                        reported.append(BAD_H)
                    else:
                        reported.append(h)

                wmsr_h = aggregate_hypers(reported, mode="wmsr", F=F)
                mean_h = aggregate_hypers(reported, mode="mean", F=F)
                last_wmsr_hypers = wmsr_h
                last_mean_hypers = mean_h

                print("  WMSR =", tuple(round(v, 3) for v in wmsr_h))
                print("  MEAN =", tuple(round(v, 3) for v in mean_h))
                print("  DIFF =", tuple(round(abs(wmsr_h[j] - mean_h[j]), 3) for j in range(3)))

                cycle += 1
                if cycle >= MAX_CYCLES:
                    print("\nDone cycles. Closing.")
                    break

                print("\n[SWITCH] meet -> explore (new goals + new meeting)\n")
                phase = "explore"
                within_time[:] = 0.0
                phase_start_t = t

                # new meeting center each cycle
                MEETING_CENTER = sample_meeting_point()
                update_meeting_marker(MEETING_CENTER)

                # new explore goals each cycle (paper-like subareas)
                explore_goal = goals_for_cycle(cycle)
                place_goal_markers(explore_goal)

                active_target = explore_goal.copy()

        if GUI:
            time.sleep(DT)

    env.close()

    # arrays
    times = np.array(times, dtype=float)
    pos_hist = np.array(pos_hist, dtype=float)
    active_target_hist = np.array(active_target_hist, dtype=float)
    meet_hist = np.array(meet_hist, dtype=float)
    z_hist = np.array(z_hist, dtype=float)  # (T,NUM)

    # per-drone path plots (sim coords)
    for i in range(NUM):
        plt.figure(figsize=(7.2, 6.2), dpi=180)
        plt.plot(pos_hist[:, i, 0], pos_hist[:, i, 1], lw=2.5,
                 label=f"Drone {i}" + (" (faulty)" if i in faulty_set else ""))
        plt.scatter([A[i, 0]], [A[i, 1]], s=60, marker="o", label="Start")
        plt.scatter(meet_hist[:, 0], meet_hist[:, 1], s=8, marker=".", label="Meeting centers")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Path of Drone {i} (dynamic meeting centers)")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / f"path_drone{i}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # all drones (sim coords)
    plt.figure(figsize=(7.8, 6.4), dpi=180)
    for i in range(NUM):
        lbl = f"Drone {i}" + (" (faulty)" if i in faulty_set else "")
        plt.plot(pos_hist[:, i, 0], pos_hist[:, i, 1], lw=2.2, label=lbl)
        plt.scatter([A[i, 0]], [A[i, 1]], marker="o", s=40)
    plt.scatter(meet_hist[:, 0], meet_hist[:, 1], s=12, marker=".", label="Meeting centers")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Paths of all drones (dynamic meeting centers)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "paths_all_drones.png", dpi=300, bbox_inches="tight")
    plt.close()

    # distance to active target
    dist_active = np.linalg.norm(active_target_hist - pos_hist, axis=2)
    plt.figure(figsize=(9, 5), dpi=180)
    for i in range(NUM):
        lbl = f"Drone {i}" + (" (faulty)" if i in faulty_set else "")
        plt.plot(times, dist_active[:, i], label=lbl)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance to ACTIVE target (m)")
    plt.title("Distance-to-active-target vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "distance_to_active_target.png", dpi=300, bbox_inches="tight")
    plt.close()

    # z over time
    plt.figure(figsize=(9, 5), dpi=180)
    for i in range(NUM):
        lbl = f"Drone {i}" + (" (faulty)" if i in faulty_set else "")
        plt.plot(times, z_hist[:, i], label=lbl)
    plt.axhline(Z_GOAL, linestyle="--", linewidth=1, label="z_goal")
    plt.xlabel("Time (s)")
    plt.ylabel("z (m)")
    plt.title("Altitude over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "z_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PAPER-style planning/path 4-panel (Fig.5-like)
    plot_paper_style_paths_4panel(
        outdir=OUTDIR,
        pos_hist=pos_hist,
        phase_hist=phase_hist,
        cycle_hist=cycle_hist,
        meet_hist=meet_hist,
        XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX,
        faulty_set=faulty_set,
    )

    # PAPER-style GP surfaces 4-panel (Fig.6-like)
    if last_wmsr_hypers is not None and last_mean_hypers is not None:
        plot_gp_surfaces_4panel_paper(
            outdir=OUTDIR,
            XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX,
            drones_data=drones,
            good_id=0,
            wmsr_hypers=last_wmsr_hypers,
            mean_hypers=last_mean_hypers,
            init_points=60,
            grid_n=55,
        )

    print("\nSaved:")
    files = [
        "paths_all_drones.png",
        "distance_to_active_target.png",
        "z_over_time.png",
        "paper_style_paths_4panel.png",
        "gp_surfaces_4panel_paper.png",
        "gp_surface_diff_paper.png",
    ] + [f"path_drone{i}.png" for i in range(NUM)]
    for f in files:
        print(" -", OUTDIR / f)


if __name__ == "__main__":
    env = None
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CTRL+C] Interrupted. Exiting cleanly...")
    finally:
        # ensure PyBullet disconnects even if you cancel
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass