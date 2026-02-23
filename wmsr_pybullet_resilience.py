#!/usr/bin/env python3
"""
W-MSR + GP hyperparameter sharing pipeline (PyBullet drones)

PIPELINE (REPEATS):
(1) Start: all 3 drones at SAME point
(2) Explore: go to exploration targets, collect measurements
(3) Meet: go to meeting point, HOLD until all reached
(4) Share: fit local GP -> hyperparams -> W-MSR fuse
(5) Replan: use fused GP to pick new exploration targets (high variance)
(6) Explore again

Works with gym-pybullet-drones version differences via constructor adapter.

Outputs:
- pipeline_paths.png
- fused_variance_map_cycleX.png (optional)
"""

import os
import time
import math
import inspect
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

try:
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
except Exception:
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Optional enums (depend on version)
try:
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType
except Exception:
    ObservationType, ActionType = None, None


# -----------------------------
# Robust env constructor
# -----------------------------
def _sig_has(name: str, sig: inspect.Signature) -> bool:
    return name in sig.parameters

def make_ctrl_aviary(num_drones: int, gui: bool, initial_xyzs: np.ndarray):
    sig = inspect.signature(CtrlAviary.__init__)
    kwargs = {}

    if _sig_has("drone_model", sig): kwargs["drone_model"] = DroneModel.CF2X
    if _sig_has("num_drones", sig): kwargs["num_drones"] = num_drones
    if _sig_has("physics", sig): kwargs["physics"] = Physics.PYB
    if _sig_has("gui", sig): kwargs["gui"] = gui
    if _sig_has("record", sig): kwargs["record"] = False
    if _sig_has("obstacles", sig): kwargs["obstacles"] = False
    if _sig_has("user_debug_gui", sig): kwargs["user_debug_gui"] = gui

    if _sig_has("initial_xyzs", sig):
        kwargs["initial_xyzs"] = initial_xyzs
    elif _sig_has("initial_xyz", sig):
        kwargs["initial_xyz"] = initial_xyzs

    if _sig_has("initial_rpys", sig):
        kwargs["initial_rpys"] = np.zeros((num_drones, 3))

    # Frequencies (best-effort)
    if _sig_has("pyb_freq", sig): kwargs["pyb_freq"] = 240
    if _sig_has("ctrl_freq", sig): kwargs["ctrl_freq"] = 48
    if _sig_has("freq", sig): kwargs["freq"] = 48
    if _sig_has("aggregate_phy_steps", sig): kwargs["aggregate_phy_steps"] = 5

    # Obs/act types (only if supported)
    if ObservationType is not None:
        if _sig_has("obs", sig): kwargs["obs"] = ObservationType.KIN
        if _sig_has("observation_type", sig): kwargs["observation_type"] = ObservationType.KIN

    if ActionType is not None:
        if _sig_has("act", sig): kwargs["act"] = ActionType.RPM
        if _sig_has("action_type", sig): kwargs["action_type"] = ActionType.RPM

    return CtrlAviary(**kwargs)


# -----------------------------
# State reading + control actions
# -----------------------------
def read_states(env, obs, N):
    states = []
    for i in range(N):
        try:
            s = env._getDroneStateVector(i)
            pos = np.array(s[0:3])
            quat = np.array(s[3:7])
            rpy = np.array(s[7:10]) if len(s) >= 10 else np.zeros(3)
            vel = np.array(s[10:13]) if len(s) >= 13 else np.zeros(3)
            ang_vel = np.array(s[13:16]) if len(s) >= 16 else np.zeros(3)
        except Exception:
            # fallback: assume obs[i][0:3] is pos
            pos = np.array(obs[i][0:3])
            quat = np.array([1,0,0,0], float)
            rpy = np.zeros(3)
            vel = np.zeros(3)
            ang_vel = np.zeros(3)

        states.append({"pos":pos, "quat":quat, "rpy":rpy, "vel":vel, "ang_vel":ang_vel})
    return states

def pid_rpm_action(ctrls, states, targets_xyz, dt):
    N = len(ctrls)
    action = np.zeros((N,4), dtype=np.float32)

    for i in range(N):
        rpm, _, _ = ctrls[i].computeControl(
            control_timestep=dt,
            cur_pos=states[i]["pos"],
            cur_quat=states[i]["quat"],
            cur_vel=states[i]["vel"],
            cur_ang_vel=states[i]["ang_vel"],
            target_pos=targets_xyz[i],
            target_rpy=np.array([0.0,0.0,0.0])
        )
        action[i,:] = rpm
    return action

def fly_hold_targets(env, ctrls, targets_xyz, reached_tol_xy=0.25, max_steps=2500, gui=False, label="[fly]"):
    """
    HOLD targets until all drones reach XY tolerance (hard guarantee unless timeout).
    Returns reached(bool), trajectory list of positions.
    """
    N = targets_xyz.shape[0]
    dt = 1.0/48.0
    try:
        if hasattr(env, "CTRL_FREQ"):
            dt = 1.0/float(env.CTRL_FREQ)
    except Exception:
        pass

    traj = []
    obs, _ = env.reset()

    for step in range(max_steps):
        states = read_states(env, obs, N)
        pos = np.stack([st["pos"] for st in states], axis=0)
        traj.append(pos.copy())

        dxy = np.linalg.norm(pos[:,0:2] - targets_xyz[:,0:2], axis=1)
        max_dist = float(np.max(dxy))
        min_dist = float(np.min(dxy))

        if step % 100 == 0:
            print(f"{label} step={step:4d} max_dist={max_dist:.3f} min_dist={min_dist:.3f} targets={targets_xyz[0,0:2]}")

        if max_dist <= reached_tol_xy:
            return True, traj

        action = pid_rpm_action(ctrls, states, targets_xyz, dt)
        obs, reward, terminated, truncated, info = env.step(action)

        if gui:
            time.sleep(dt)
        if terminated or truncated:
            break

    return False, traj


# -----------------------------
# Synthetic field + sensing
# -----------------------------
def true_field(xy):
    """Synthetic smooth 2D field; think 'temperature'."""
    x, y = xy
    return (np.sin(1.2*x)*np.cos(1.0*y) + 0.35*np.cos(2.0*x + 0.7*y))

def sense_field_at_positions(pos_xyz, noise_std=0.02, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    ys = []
    for p in pos_xyz:
        val = true_field(p[0:2]) + rng.normal(0.0, noise_std)
        ys.append(val)
    return np.array(ys, dtype=float)


# -----------------------------
# GP fit + hyperparam extraction
# -----------------------------
def fit_gp_and_get_hyperparams(X, y):
    """
    Fit GP and return hyperparams:
      amp (signal std), length_scale, noise_std
    """
    # kernel = C * RBF + White
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2)) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0
    )
    gp.fit(X, y)

    # Extract
    k = gp.kernel_
    # Typical structure: (C*RBF) + White
    # k.k1 is (C*RBF), k.k2 is White
    amp = math.sqrt(float(k.k1.k1.constant_value))  # signal std
    ls = float(k.k1.k2.length_scale)
    noise_std = math.sqrt(float(k.k2.noise_level))

    return gp, np.array([amp, ls, noise_std], dtype=float)


# -----------------------------
# W-MSR (trimmed mean) fusion for hyperparams
# -----------------------------
def wmsr_fuse(hparams_matrix, F=1):
    """
    hparams_matrix: shape (N, D) where D=3 (amp, ls, noise)
    W-MSR rule per dimension:
      - sort values
      - trim F lowest and F highest
      - average remaining
    """
    N, D = hparams_matrix.shape
    fused = np.zeros(D, dtype=float)

    for d in range(D):
        vals = np.sort(hparams_matrix[:, d])
        if 2*F >= N:
            # can't trim, fallback to median
            fused[d] = np.median(vals)
        else:
            fused[d] = np.mean(vals[F:N-F])

    return fused


# -----------------------------
# Replan using fused GP: pick high-variance points
# -----------------------------
def build_gp_from_fused_hparams(fused, X, y):
    """
    Build a GP using *fixed* fused hyperparams (no re-optimization).
    """
    amp, ls, noise_std = fused
    kernel = ConstantKernel(amp**2) * RBF(length_scale=ls) + WhiteKernel(noise_level=noise_std**2)

    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
    gp.fit(X, y)
    return gp

def pick_high_variance_targets(gp, bounds=(-2,2), grid_n=35, k=3, min_sep=0.6, rng=None):
    """
    Pick k targets in XY at highest predictive std, with separation.
    """
    rng = np.random.default_rng() if rng is None else rng
    xs = np.linspace(bounds[0], bounds[1], grid_n)
    ys = np.linspace(bounds[0], bounds[1], grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)

    _, std = gp.predict(pts, return_std=True)
    order = np.argsort(-std)  # descending
    chosen = []
    for idx in order:
        p = pts[idx]
        if all(np.linalg.norm(p - c) >= min_sep for c in chosen):
            chosen.append(p)
        if len(chosen) >= k:
            break

    chosen = np.array(chosen, float)
    return chosen, (xs, ys, std.reshape(grid_n, grid_n))


# -----------------------------
# Plotting
# -----------------------------
def plot_paths(paths_xyz, meeting_point, outpath, title):
    """
    paths_xyz: list of arrays (T,N,3) or one concatenated array
    """
    if isinstance(paths_xyz, list):
        P = np.concatenate(paths_xyz, axis=0)
    else:
        P = paths_xyz

    T, N, _ = P.shape
    ds = max(1, T // 200)  # downsample for clean plot
    P = P[::ds]

    plt.figure(figsize=(8,7))
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.axis("equal")
    plt.xlim(-2,2); plt.ylim(-2,2)

    # Background: true field
    xs = np.linspace(-2,2,60)
    ys = np.linspace(-2,2,60)
    X, Y = np.meshgrid(xs, ys)
    Z = np.sin(1.2*X)*np.cos(1.0*Y) + 0.35*np.cos(2.0*X + 0.7*Y)
    plt.imshow(Z, extent=[-2,2,-2,2], origin="lower", alpha=0.85)

    colors = ["k","g","tab:blue"]

    # Start
    plt.scatter([P[0,0,0]],[P[0,0,1]], s=120, facecolor="w", edgecolor="k", label="Start")

    # Meeting point
    plt.scatter([meeting_point[0]],[meeting_point[1]], s=140, marker="s", edgecolor="k", label="Meeting")

    for i in range(N):
        plt.plot(P[:,i,0], P[:,i,1], linewidth=2.5, color=colors[i], label=f"drone{i}")

    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[plot] saved: {outpath}")

def plot_variance_map(xs, ys, std_grid, targets, outpath, title):
    plt.figure(figsize=(7,6))
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.imshow(std_grid, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", aspect="equal")
    plt.scatter(targets[:,0], targets[:,1], s=120, facecolor="none", edgecolor="w", linewidth=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[plot] saved: {outpath}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    # ===== Settings =====
    N = 3
    gui = True
    Z = 1.0
    rng = np.random.default_rng(7)

    START_XYZ = np.array([0.0, 0.0, Z])              # SAME start for all
    MEETING_XY = np.array([1.2, 1.2], dtype=float)   # meeting point
    meeting_xyz = np.array([MEETING_XY[0], MEETING_XY[1], Z], dtype=float)

    # Fault model
    faulty_set = {2}   # drone index that lies about hyperparams
    F = 1              # W-MSR trims 1 lowest & 1 highest (works for N=3)

    # Exploration cycles
    CYCLES = 2
    EXPLORATION_POINTS_PER_CYCLE = 3

    # ===== Init env =====
    initial_xyzs = np.vstack([START_XYZ for _ in range(N)])
    env = make_ctrl_aviary(N, gui, initial_xyzs)
    ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(N)]

    # Data buffers per drone
    X_data = [ [] for _ in range(N) ]  # list of xy
    y_data = [ [] for _ in range(N) ]  # list of scalar

    all_trajs = []

    # ===== Phase 0: stabilize hover at start =====
    print("[phase0] takeoff/stabilize at start...")
    start_targets = initial_xyzs.copy()
    reached, traj0 = fly_hold_targets(env, ctrls, start_targets, reached_tol_xy=0.35, max_steps=600, gui=gui, label="[start]")
    all_trajs.append(np.array(traj0))

    # collect a first measurement at start
    pos0 = traj0[-1]
    meas0 = sense_field_at_positions(pos0, noise_std=0.02, rng=rng)
    for i in range(N):
        X_data[i].append(pos0[i,0:2].copy())
        y_data[i].append(float(meas0[i]))

    fused_hparams = None

    # ===== Repeat cycles =====
    for cyc in range(1, CYCLES+1):
        print(f"\n================= CYCLE {cyc} =================")

        # ---- (1) Exploration targets ----
        # First cycle: random exploration; later cycles: high-variance targets from fused GP
        if fused_hparams is None:
            # random but deterministic exploration
            targets_xy = rng.uniform(-1.8, 1.8, size=(EXPLORATION_POINTS_PER_CYCLE, 2))
        else:
            # Build a global dataset for fused GP planning (use all non-faulty data OR all data)
            Xg = np.vstack([np.array(X_data[i]) for i in range(N)])
            yg = np.hstack([np.array(y_data[i]) for i in range(N)])
            gp_fused = build_gp_from_fused_hparams(fused_hparams, Xg, yg)
            targets_xy, (xs, ys, std_grid) = pick_high_variance_targets(gp_fused, bounds=(-2,2), grid_n=35, k=EXPLORATION_POINTS_PER_CYCLE, min_sep=0.8, rng=rng)

            plot_variance_map(xs, ys, std_grid, targets_xy,
                              outpath=f"fused_variance_map_cycle{cyc}.png",
                              title=f"Fused GP std map (cycle {cyc})")

        # Assign exploration points to drones (simple: each point becomes a waypoint all drones visit in different order)
        # We create per-drone target sequence by rotating order
        per_drone_sequences = []
        for i in range(N):
            seq = np.roll(targets_xy, shift=i, axis=0)
            per_drone_sequences.append(seq)

        # ---- (2) Explore: go waypoint by waypoint ----
        print("[explore] moving and sensing...")
        for wp_idx in range(EXPLORATION_POINTS_PER_CYCLE):
            targets_xyz = np.zeros((N,3), dtype=float)
            for i in range(N):
                targets_xyz[i,0:2] = per_drone_sequences[i][wp_idx]
                targets_xyz[i,2] = Z

            reached, traj = fly_hold_targets(env, ctrls, targets_xyz,
                                            reached_tol_xy=0.30, max_steps=2200,
                                            gui=gui, label=f"[explore{cyc}.{wp_idx}]")
            all_trajs.append(np.array(traj))

            # sense + store data at end of waypoint
            pos = traj[-1]
            meas = sense_field_at_positions(pos, noise_std=0.02, rng=rng)
            for i in range(N):
                X_data[i].append(pos[i,0:2].copy())
                y_data[i].append(float(meas[i]))

        # ---- (3) Meet: HARD hold at meeting point until reached ----
        print("[meet] going to meeting point and HOLD...")
        meet_targets = np.vstack([meeting_xyz for _ in range(N)])
        reached, traj_meet = fly_hold_targets(env, ctrls, meet_targets,
                                             reached_tol_xy=0.25, max_steps=3500,
                                             gui=gui, label=f"[meet{cyc}]")
        print(f"[meet] reached={reached}")
        all_trajs.append(np.array(traj_meet))

        # ---- (4) Share hyperparams at meeting: local GP fit -> WMSR fuse ----
        print("[share] fitting local GPs and fusing hyperparams with W-MSR...")
        local_hparams = np.zeros((N,3), dtype=float)

        for i in range(N):
            Xi = np.array(X_data[i])
            yi = np.array(y_data[i])
            gp_i, h_i = fit_gp_and_get_hyperparams(Xi, yi)

            # Faulty agent lies about hyperparams
            if i in faulty_set:
                # exaggerate amp and length-scale, mess with noise
                h_i = np.array([h_i[0]*5.0, h_i[1]*8.0, max(1e-3, h_i[2]*0.1)], dtype=float)
                print(f"  drone{i} is FAULTY -> sending bad h={h_i}")
            else:
                print(f"  drone{i} honest -> h={h_i}")

            local_hparams[i,:] = h_i

        fused_hparams = wmsr_fuse(local_hparams, F=F)
        print(f"[WMSR] fused_hparams = (amp={fused_hparams[0]:.3f}, ls={fused_hparams[1]:.3f}, noise={fused_hparams[2]:.6f})")

        # ---- (5) Explore again after meeting (pipeline requirement) ----
        # We'll do a short post-meeting exploration hop to prove "meet -> fuse -> explore again"
        print("[post] short exploration after fusion...")
        post_xy = rng.uniform(-1.6, 1.6, size=(N,2))
        post_targets = np.hstack([post_xy, Z*np.ones((N,1))])
        reached, traj_post = fly_hold_targets(env, ctrls, post_targets,
                                             reached_tol_xy=0.30, max_steps=2000,
                                             gui=gui, label=f"[post{cyc}]")
        all_trajs.append(np.array(traj_post))

        # sense + store
        pos = traj_post[-1]
        meas = sense_field_at_positions(pos, noise_std=0.02, rng=rng)
        for i in range(N):
            X_data[i].append(pos[i,0:2].copy())
            y_data[i].append(float(meas[i]))

    # ===== Close env =====
    try:
        env.close()
    except Exception:
        pass

    # ===== Plot paths =====
    P = np.concatenate(all_trajs, axis=0)
    plot_paths(P, MEETING_XY, outpath="pipeline_paths.png",
               title="W-MSR pipeline: start same point → explore → meet → WMSR fuse → explore again")

    print("\n[done] outputs: pipeline_paths.png (+ variance maps if fused cycles)")


if __name__ == "__main__":
    main()