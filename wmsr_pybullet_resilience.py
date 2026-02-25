# scripts/four_drones_wmsr.py
# FULL WORKING CODE
# - 4 drones in Gym-PyBullet (CtrlAviary) using built-in PID (DSLPIDControl)
# - PID outputs motor RPMs -> env.step(rpms)
# - Adds a simple "meeting + WMSR fusion" pipeline (share GP hyperparameters only)
# - Logs + saves clear plots:
#     - wmsr_distance.png  (distance-to-goal vs time)
#     - wmsr_x.png         (x(t) vs time)
#     - wmsr_xy.png        (XY trajectories with A/B markers)
#     - planning_paths_4panel.png  (robot1/2/3 paths + replanning after meeting)
#
# Run:
#   (.venv) python scripts/four_drones_wmsr.py

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel


# -----------------------------
# Synthetic scalar field (virtual sensor)
# yi = f(x,y) + noise (+ attack for faulty)
# -----------------------------
def field_f(x: float, y: float) -> float:
    # Smooth, nontrivial field (like temperature/gas concentration)
    return (
        1.2 * np.sin(0.8 * x)
        + 0.9 * np.cos(0.6 * y)
        + 0.6 * np.sin(0.35 * x * y)
    )


# -----------------------------
# GP helpers (same idea as your python mapping code)
# Each drone fits GP locally and extracts hypers (amp, ls, noi)
# -----------------------------
def build_gp(amp=1.0, ls=1.0, noi=0.2, *, optimize=True, seed=0):
    kernel = (
        C(amp, (1e-3, 1e3))
        * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=noi, noise_level_bounds=(1e-6, 1e2))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=3 if optimize else 0,
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
# Plot helpers (paper-ish paths)
# -----------------------------
def field_grid(xmin=0.0, xmax=2.0, ymin=0.0, ymax=1.2, n=60):
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    XX, YY = np.meshgrid(xs, ys)
    Z = field_f(XX, YY)
    return xs, ys, Z


def meeting_area_points(center_xy, w=0.35, h=0.35, step=0.06):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    xs = np.arange(cx - w / 2, cx + w / 2 + 1e-9, step)
    ys = np.arange(cy - h / 2, cy + h / 2 + 1e-9, step)
    XX, YY = np.meshgrid(xs, ys)
    return np.c_[XX.ravel(), YY.ravel()]


def compress_path(P, keep_every=8):
    P = np.asarray(P, dtype=float)
    if P.shape[0] <= 2:
        return P
    idx = np.arange(0, P.shape[0], keep_every, dtype=int)
    if idx[-1] != P.shape[0] - 1:
        idx = np.append(idx, P.shape[0] - 1)
    return P[idx]


# -----------------------------
# Env state compatibility
# -----------------------------
def get_state_vector(env, i: int):
    if hasattr(env, "getDroneStateVector"):
        return env.getDroneStateVector(i)
    return env._getDroneStateVector(i)


# -----------------------------
# MAIN
# -----------------------------
def main():
    OUTDIR = Path(".")
    OUTDIR.mkdir(exist_ok=True)

    # ---------- SIM SETTINGS ----------
    NUM = 4
    CTRL_FREQ = 240
    DT = 1.0 / CTRL_FREQ
    DURATION_SEC = 20.0
    STEPS = int(DURATION_SEC * CTRL_FREQ)

    GUI = True
    DRONE_MODEL = DroneModel.CF2X
    PHY = Physics.PYB

    # Reach/settle detection
    TOL = 0.12          # meters
    SETTLE_SEC = 1.0    # seconds within tolerance before "arrived"

    # ---------- A (starts) and B (goals) ----------
    A = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.4, 1.0],
            [0.4, 0.0, 1.0],
            [0.4, 0.4, 1.0],
        ],
        dtype=np.float32,
    )

    B = np.array(
        [
            [1.0, 0.5, 1.2],
            [1.0, 0.9, 1.2],
            [1.6, 0.5, 1.6],
            [1.2, 0.9, 1.2],
        ],
        dtype=np.float32,
    )

    TARGET_RPY = np.zeros((NUM, 3), dtype=np.float32)  # [roll,pitch,yaw] targets
    TARGET_RPY[:, 2] = 0.0

    # ---------- MEETING SETTINGS ----------
    # We force a rendezvous at this XY and do WMSR fusion there.
    MEETING_POINT = np.array([0.75, 0.60, 1.20], dtype=np.float32)
    MEETING_EVERY_SEC = 6.0
    MEETING_EVERY_STEPS = int(MEETING_EVERY_SEC * CTRL_FREQ)
    MEETING_RADIUS = 0.15  # meters

    # ---------- WMSR / FAULT MODEL ----------
    # With N=4, a reasonable trim is F=1 (drop smallest + largest, average middle 2)
    MODE = "wmsr"
    F = 1

    # Choose one faulty agent (biasing its measurement)
    faulty_set = {2}          # change if you want
    attack_bias = 2.5         # adds spatial bias
    meas_noise_std = 0.05     # all drones noise
    faulty_extra_noise = 0.15 # extra noisy for faulty

    print("4-Drone PID A -> B demo + meeting/WMSR fusion (PID outputs RPMs, env takes RPMs).")
    print("A:\n", A)
    print("B:\n", B)
    print(f"tol={TOL}m duration={DURATION_SEC}s freq={CTRL_FREQ}Hz")
    print(f"meeting every {MEETING_EVERY_SEC}s at {MEETING_POINT[:2].tolist()}  (radius {MEETING_RADIUS}m)")
    print(f"faulty agents = {sorted(list(faulty_set))} | fusion={MODE} F={F}")
    print("Tip: close the PyBullet window or press Ctrl+C to stop.\n")

    # ---------- ENV ----------
    env = CtrlAviary(
        drone_model=DRONE_MODEL,
        num_drones=NUM,
        initial_xyzs=A,
        initial_rpys=TARGET_RPY,
        physics=PHY,
        gui=GUI,
        record=False,
        ctrl_freq=CTRL_FREQ,  # <-- correct for CtrlAviary
    )
    obs, info = env.reset()

    # ---------- CONTROLLERS ----------
    ctrls = [DSLPIDControl(drone_model=DRONE_MODEL) for _ in range(NUM)]

    # ---------- LOCAL GP STATE PER DRONE ----------
    # Each drone stores (xi, yi) in 2D (x,y) and fits GP
    drones = []
    for i in range(NUM):
        gp = build_gp(optimize=True, seed=10 + i)
        X = np.empty((0, 2), dtype=float)
        y = np.empty((0,), dtype=float)
        drones.append(
            {
                "id": i,
                "gp": gp,
                "X": X,
                "y": y,
            }
        )

    # ---------- LOGS ----------
    times = np.zeros(STEPS, dtype=float)
    pos_hist = np.zeros((STEPS, NUM, 3), dtype=float)
    err_hist = np.zeros((STEPS, NUM), dtype=float)

    # For “paper” path plots
    paths_xy = [[] for _ in range(NUM)]
    meet_indices = []  # index in path timeline where meeting happened

    # Arrival bookkeeping
    within_tol_time = np.zeros(NUM, dtype=float)
    arrived = np.zeros(NUM, dtype=bool)

    # Meeting scheduling
    next_meeting_step = MEETING_EVERY_STEPS

    # Active target per drone (switches to meeting point during meeting phase)
    active_target = B.copy()
    in_meeting_mode = False

    # Visual markers
    p.addUserDebugText("MEETING", [float(MEETING_POINT[0]), float(MEETING_POINT[1]), float(MEETING_POINT[2])],
                       textColorRGB=[1, 0, 1], textSize=1.2, lifeTime=0)
    for i in range(NUM):
        p.addUserDebugText(f"A{i}", [float(A[i,0]), float(A[i,1]), float(A[i,2])],
                           textColorRGB=[0, 0.7, 1], textSize=1.0, lifeTime=0)
        p.addUserDebugText(f"B{i}", [float(B[i,0]), float(B[i,1]), float(B[i,2])],
                           textColorRGB=[0, 1, 0], textSize=1.0, lifeTime=0)

    # init paths
    for i in range(NUM):
        paths_xy[i].append([float(A[i, 0]), float(A[i, 1])])

    try:
        for k in range(STEPS):
            t = k * DT
            times[k] = t

            # Decide if we should enter meeting mode
            if (not in_meeting_mode) and (k == next_meeting_step):
                in_meeting_mode = True
                active_target = np.tile(MEETING_POINT.reshape(1, 3), (NUM, 1)).astype(np.float32)
                within_tol_time[:] = 0.0
                arrived[:] = False
                # schedule next meeting later (after this one completes)
                next_meeting_step += MEETING_EVERY_STEPS

            # Read states and compute RPM for all drones
            rpms = np.zeros((NUM, 4), dtype=np.float32)

            for i in range(NUM):
                state = np.array(get_state_vector(env, i), dtype=np.float32)

                pos = state[0:3]
                quat = state[3:7]
                vel = state[10:13] if state.shape[0] >= 13 else np.zeros(3, dtype=np.float32)
                ang_vel = state[13:16] if state.shape[0] >= 16 else np.zeros(3, dtype=np.float32)

                # PID -> RPMs
                out = ctrls[i].computeControl(
                    control_timestep=DT,
                    cur_pos=pos,
                    cur_quat=quat,
                    cur_vel=vel,
                    cur_ang_vel=ang_vel,
                    target_pos=active_target[i],
                    target_rpy=TARGET_RPY[i],
                )
                rpm = out[0] if isinstance(out, (tuple, list)) else out
                rpms[i, :] = np.array(rpm, dtype=np.float32).reshape(4,)

                # Logs
                pos_hist[k, i, :] = pos
                err = active_target[i] - pos
                err_norm = float(np.linalg.norm(err))
                err_hist[k, i] = err_norm

                paths_xy[i].append([float(pos[0]), float(pos[1])])

                # Synthetic measurement yi at current (x,y)
                x, y = float(pos[0]), float(pos[1])
                true_val = float(field_f(x, y))

                # Add noise (+ attack if faulty)
                noise = np.random.normal(0.0, meas_noise_std)
                if i in faulty_set:
                    attack = attack_bias * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
                    noise += np.random.normal(0.0, faulty_extra_noise)
                    meas = true_val + float(attack) + float(noise)
                else:
                    meas = true_val + float(noise)

                # Store (xi, yi)
                drones[i]["X"] = np.vstack([drones[i]["X"], np.array([[x, y]], dtype=float)])
                drones[i]["y"] = np.append(drones[i]["y"], meas)

            # Step env
            obs, reward, terminated, truncated, info = env.step(rpms)

            # Meeting completion check (all within meeting radius)
            if in_meeting_mode:
                all_within = True
                for i in range(NUM):
                    if err_hist[k, i] <= MEETING_RADIUS:
                        within_tol_time[i] += DT
                    else:
                        within_tol_time[i] = 0.0

                    if within_tol_time[i] >= SETTLE_SEC:
                        arrived[i] = True
                    all_within = all_within and arrived[i]

                if all_within:
                    # ---- MEET & SHARE HYPERS ----
                    hypers = []
                    for i in range(NUM):
                        gp = drones[i]["gp"]
                        X = drones[i]["X"]
                        y = drones[i]["y"]
                        if X.shape[0] >= 5:
                            gp.fit(X, y)
                            hypers.append(extract_hypers(gp))
                        else:
                            # not enough points -> default-ish
                            hypers.append((1.0, 1.0, 0.2))

                    agg_amp, agg_ls, agg_noi = aggregate_hypers(hypers, mode=MODE, F=F)

                    print(
                        f"[MEETING @ t={t:.2f}s] shared hypers (amp,ls,noi) -> fused: "
                        f"amp={agg_amp:.3f} ls={agg_ls:.3f} noi={agg_noi:.3f}"
                    )

                    # Rebuild all GPs with fused hypers (hyperparameter consensus)
                    for i in range(NUM):
                        drones[i]["gp"] = build_gp(
                            amp=agg_amp,
                            ls=agg_ls,
                            noi=agg_noi,
                            optimize=False,     # keep them fixed after fusion for stability
                            seed=10 + i,
                        )

                    # mark split for “replanning after meeting” plot
                    meet_indices.append(len(paths_xy[0]) - 1)

                    # Exit meeting mode -> resume goal B
                    in_meeting_mode = False
                    active_target = B.copy()
                    within_tol_time[:] = 0.0
                    arrived[:] = False

            # Keep real-time feeling
            if GUI:
                time.sleep(DT)

    except KeyboardInterrupt:
        print("Stopped by user.")

    env.close()

    # -----------------------------
    # PLOTS (clear, no overlap confusion)
    # -----------------------------
    # Distance-to-goal vs time (to FINAL goal B, not meeting)
    dist_to_B = np.zeros((STEPS, NUM), dtype=float)
    for k in range(STEPS):
        for i in range(NUM):
            pos = pos_hist[k, i, :]
            dist_to_B[k, i] = float(np.linalg.norm(B[i] - pos))

    plt.figure(figsize=(9, 5), dpi=180)
    for i in range(NUM):
        plt.plot(times, dist_to_B[:, i], label=f"Drone {i}")
    plt.axhline(TOL, linestyle="--", linewidth=1, label="tolerance")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance to goal ||e|| (m)")
    plt.title("4 drones: distance-to-goal vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "wmsr_distance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # x(t) vs time (each drone)
    plt.figure(figsize=(9, 5), dpi=180)
    for i in range(NUM):
        plt.plot(times, pos_hist[:, i, 0], label=f"Drone {i} x(t)")
        plt.axhline(float(B[i, 0]), linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("x position (m)")
    plt.title("4 drones: x(t) vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "wmsr_x.png", dpi=300, bbox_inches="tight")
    plt.close()

    # XY trajectories with A and B markers
    plt.figure(figsize=(7.5, 6), dpi=180)
    for i in range(NUM):
        plt.plot(pos_hist[:, i, 0], pos_hist[:, i, 1], label=f"Drone {i}")
        plt.scatter([A[i, 0]], [A[i, 1]], marker="o")
        plt.scatter([B[i, 0]], [B[i, 1]], marker="x")
    plt.scatter([MEETING_POINT[0]], [MEETING_POINT[1]], marker="s", s=90, label="Meeting")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("4 drones: XY trajectories (A=o, B=x)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "wmsr_xy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Paper-style 4 panel (robot 1/2/3 + replanning after meeting) ----
    xs, ys, Zbg = field_grid(
        xmin=min(A[:, 0].min(), B[:, 0].min()) - 0.2,
        xmax=max(A[:, 0].max(), B[:, 0].max()) + 0.2,
        ymin=min(A[:, 1].min(), B[:, 1].min()) - 0.2,
        ymax=max(A[:, 1].max(), B[:, 1].max()) + 0.2,
        n=70,
    )
    meet_xy = np.array([MEETING_POINT[0], MEETING_POINT[1]], dtype=float)
    meet_pts = meeting_area_points(meet_xy, w=0.35, h=0.35, step=0.06)

    split_idx = meet_indices[0] if len(meet_indices) else None

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=180)
    panel_titles = ["(a) Path for robot 1", "(b) Path for robot 2", "(c) Path for robot 3", "(d) Re-planning after meeting"]

    def _panel_bg(ax):
        ax.pcolormesh(xs, ys, Zbg, shading="nearest")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(meet_pts[:, 0], meet_pts[:, 1], s=10, c="red", marker=".", label="Meeting Area")
        ax.scatter([meet_xy[0]], [meet_xy[1]], s=80, marker="s", c="pink", edgecolors="k", label="Meeting Location")

    # panels a/b/c for drones 1,2,3 (ids 0,1,2)
    for j, rid in enumerate([0, 1, 2]):
        ax = axes[j]
        _panel_bg(ax)
        ax.set_title(panel_titles[j])

        ax.scatter([A[rid, 0]], [A[rid, 1]], s=60, c="white", edgecolors="k", label="Start")
        ax.scatter([B[rid, 0]], [B[rid, 1]], s=60, c="lime", edgecolors="k", marker="x", label="Goal")

        P = np.array(paths_xy[rid], dtype=float)
        if split_idx is not None:
            P = P[: split_idx + 1]
        P = compress_path(P, keep_every=10)
        ax.plot(P[:, 0], P[:, 1], lw=3.0, color="black", label=f"robot{rid+1}")
        ax.scatter(P[:, 0], P[:, 1], s=18, marker="s", c="pink", alpha=0.9)

        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    # panel d: post-meeting segments for all drones
    ax = axes[3]
    _panel_bg(ax)
    ax.set_title(panel_titles[3])
    colors = ["black", "green", "blue", "orange"]
    for i in range(NUM):
        P = np.array(paths_xy[i], dtype=float)
        if split_idx is not None and split_idx < len(P) - 1:
            P = P[split_idx:]
        P = compress_path(P, keep_every=14)
        ax.plot(P[:, 0], P[:, 1], lw=3.0, color=colors[i % len(colors)], label=f"robot{i+1}")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTDIR / "planning_paths_4panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved plots:")
    print(f" - {OUTDIR / 'wmsr_distance.png'}")
    print(f" - {OUTDIR / 'wmsr_x.png'}")
    print(f" - {OUTDIR / 'wmsr_xy.png'}")
    print(f" - {OUTDIR / 'planning_paths_4panel.png'}")


if __name__ == "__main__":
    main()