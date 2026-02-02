# multi_agent_resilience.py
from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel


# -----------------------------
# GP helpers
# -----------------------------
def build_gp(amp=1.0, ls=1.0, noi=1.0, *, optimize=True, seed=0):
    """
    Matérn GP with White noise.

    Key change vs your earlier version:
      - normalize_y=False  (so constant/bias corruption actually affects fitting)
      - explicit bounds so optimizer can move hypers meaningfully
    """
    kernel = (
        C(amp, (1e-3, 1e3))
        * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=noi, noise_level_bounds=(1e-6, 1e2))
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,  # IMPORTANT: avoid washing out constant bias
        n_restarts_optimizer=5 if optimize else 0,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        random_state=seed,
    )


def extract_hypers(gp):
    """
    Extract (amplitude, length_scale, noise) from fitted sklearn GP.
    """
    k = gp.kernel_
    amp = float(k.k1.k1.constant_value)
    ls = float(k.k1.k2.length_scale)
    noi = float(k.k2.noise_level)
    return amp, ls, noi


# -----------------------------
# Resilient aggregation
# -----------------------------
def wmsr(vals, F):
    """
    W-MSR trimmed mean:
      remove F smallest + F largest, then average the rest.

    Correct trimming condition:
      - trimming works when 2F < n (at least 1 value remains)
    """
    vals = sorted([float(v) for v in vals])
    n = len(vals)
    if n == 0:
        raise ValueError("wmsr got empty list.")

    # If trimming would remove all values, fall back to mean
    if 2 * F >= n:
        return float(np.mean(vals))

    return float(np.mean(vals[F : n - F]))


def aggregate(hypers_list, mode="mean", F=1):
    """
    hypers_list: list of (amp, ls, noi) across agents
    mode: 'mean' or 'wmsr'
    """
    amps = [h[0] for h in hypers_list]
    lss = [h[1] for h in hypers_list]
    nois = [h[2] for h in hypers_list]

    if mode == "mean":
        return float(np.mean(amps)), float(np.mean(lss)), float(np.mean(nois))
    if mode == "wmsr":
        return wmsr(amps, F), wmsr(lss, F), wmsr(nois, F)

    raise ValueError("mode must be 'mean' or 'wmsr'")


# -----------------------------
# Fault model (spatially inconsistent attack)
# -----------------------------
def spatial_attack(x_new, bias):
    """
    Location-dependent corruption so faulty agents learn different hypers.
    x_new: shape (1,2)
    """
    x1 = float(x_new[0, 0])
    x2 = float(x_new[0, 1])

    # Keep it smooth but clearly non-constant. Works for most normalized grid coordinates.
    # If your grid is not in [0,1], this still varies spatially.
    return float(bias * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2))


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
    T=40,
    meeting_every=10,
    F=1,
    n_faulty=1,
    faulty_set=None,
    # corruption controls
    faulty_bias=7.0,
    faulty_noise_std=3.0,
    clean_noise_std=0.0,
    seed=0,
    # meeting behavior
    freeze_after_meeting=True,
    # robust fusion at the end (makes the resilient plot very clear)
    fuse_predictions=False,
    verbose=True,
):
    """
    mode:
      - 'mean'  -> non-resilient aggregation of hypers
      - 'wmsr'  -> resilient W-MSR aggregation of hypers

    Faulty agents corrupt measurements:
      y_meas = y_true + spatial_attack(x) + N(0, faulty_noise_std)

    Returns:
      y_fused: fused prediction over grid (shape: M)
    """
    if mode not in ("mean", "wmsr"):
        raise ValueError("mode must be 'mean' or 'wmsr'")

    rng_global = np.random.default_rng(seed)
    tree = cKDTree(grid)

    # choose faulty set once (or accept caller-provided)
    if faulty_set is None:
        idxs = np.arange(N)
        rng_global.shuffle(idxs)
        faulty_set = set(idxs[:n_faulty])
    else:
        faulty_set = set(faulty_set)

    agents = []
    for i in range(N):
        X = X0.copy()
        y = y0.copy()

        used = np.zeros(grid.shape[0], dtype=bool)
        _, idx0 = tree.query(X)
        used[idx0] = True

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
            }
        )

    for t in range(1, T + 1):
        # local update + sample selection
        for a in agents:
            a["gp"].fit(a["X"], a["y"])
            _, y_std = a["gp"].predict(grid, return_std=True)

            # avoid reusing points
            y_std_masked = y_std.copy()
            y_std_masked[a["used"]] = -np.inf

            # jitter tie-break
            jitter = 1e-9 * a["rng"].standard_normal(size=y_std_masked.shape[0])
            idx_next = int(np.argmax(y_std_masked + jitter))

            x_new = grid[idx_next].reshape(1, 2)
            y_true = float(y_actual[idx_next])

            if a["faulty"]:
                attack = spatial_attack(x_new, faulty_bias)
                y_meas = y_true + attack + float(a["rng"].normal(0, faulty_noise_std))
            else:
                y_meas = y_true + float(a["rng"].normal(0, clean_noise_std))

            a["X"] = np.vstack([a["X"], x_new])
            a["y"] = np.append(a["y"], y_meas)
            a["used"][idx_next] = True

        # meeting: fuse hypers
        if t % meeting_every == 0:
            hypers = []
            for a in agents:
                a["gp"].fit(a["X"], a["y"])
                hypers.append(extract_hypers(a["gp"]))

            agg_amp, agg_ls, agg_noi = aggregate(hypers, mode=mode, F=F)

            if verbose:
                amps = np.sort([h[0] for h in hypers])
                lss = np.sort([h[1] for h in hypers])
                nois = np.sort([h[2] for h in hypers])
                print(
                    f"[{mode}] meeting t={t:02d} | faulty={sorted(list(faulty_set))}\n"
                    f"    raw amps : {np.round(amps, 3)}\n"
                    f"    raw ls   : {np.round(lss, 3)}\n"
                    f"    raw noise: {np.round(nois, 3)}\n"
                    f"    agg      : amp={agg_amp:.3f}, ls={agg_ls:.3f}, noi={agg_noi:.3f}"
                )

            # rebuild agents with fused hypers
            for i, a in enumerate(agents):
                a["gp"] = build_gp(
                    amp=agg_amp,
                    ls=agg_ls,
                    noi=agg_noi,
                    optimize=(not freeze_after_meeting),  # allow recovery if False
                    seed=seed + i,
                )

    # final predictions per agent
    preds = []
    for a in agents:
        a["gp"].fit(a["X"], a["y"])
        preds.append(a["gp"].predict(grid))
    preds = np.vstack(preds)  # shape (N, M)

    if fuse_predictions:
        # robust pointwise fusion (very visible difference)
        y_fused = np.zeros(preds.shape[1])
        for j in range(preds.shape[1]):
            if mode == "wmsr":
                y_fused[j] = wmsr(preds[:, j], F)
            else:
                y_fused[j] = float(np.mean(preds[:, j]))
        return y_fused

    # default: average agent means (kept for consistency with your old code)
    return np.mean(preds, axis=0)


# -----------------------------
# Plotting
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


def rmse(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    BASE_DIR = Path(__file__).parent
    mat = scipy.io.loadmat(BASE_DIR / "matlab_workspace.mat")
    g = mat["g"][0, 0]

    X0 = np.array(g["xi"], dtype=float)
    y0 = np.array(g["yi"], dtype=float).reshape(-1)
    grid = np.array(g["grid"], dtype=float)
    y_actual = np.array(g["yactual"], dtype=float).reshape(-1)

    GRID_RES = int(np.sqrt(grid.shape[0]))
    if GRID_RES * GRID_RES != grid.shape[0]:
        raise ValueError("Expected grid to be square (e.g., 25x25).")

    # (a) Actual
    Z_actual = y_actual.reshape(GRID_RES, GRID_RES)

    # (b) Initial GP
    gp_init = build_gp(optimize=True, seed=123)
    gp_init.fit(X0, y0)
    y_init = gp_init.predict(grid)
    Z_init = y_init.reshape(GRID_RES, GRID_RES)

    # -----------------------------
    # Settings (these will show a difference)
    # -----------------------------
    N = 7
    T = 50
    meeting_every = 10

    n_faulty = 2
    F = 2  # for N=7, trimming 2 low + 2 high leaves 3 values

    faulty_bias = 7.0
    faulty_noise_std = 3.0
    clean_noise_std = 0.0

    seed = 0

    # Pick faulty set ONCE so mean vs wmsr is apples-to-apples
    rng = np.random.default_rng(seed)
    idxs = np.arange(N)
    rng.shuffle(idxs)
    faulty_set = set(idxs[:n_faulty])
    print(f"[main] fixed faulty_set={sorted(list(faulty_set))} (N={N}, n_faulty={n_faulty}, F={F})")

    # If you want even clearer separation, set freeze_after_meeting=False
    freeze_after_meeting = True

    # This makes the resilient output extremely clear (recommended)
    fuse_predictions = True

    # (c) Non-resilient
    y_nonres = run_multi_agent(
        "mean",
        X0,
        y0,
        grid,
        y_actual,
        N=N,
        T=T,
        meeting_every=meeting_every,
        F=F,
        n_faulty=n_faulty,
        faulty_set=faulty_set,
        faulty_bias=faulty_bias,
        faulty_noise_std=faulty_noise_std,
        clean_noise_std=clean_noise_std,
        seed=seed,
        freeze_after_meeting=freeze_after_meeting,
        fuse_predictions=fuse_predictions,
        verbose=True,
    )
    Z_nonres = y_nonres.reshape(GRID_RES, GRID_RES)

    # (d) Resilient
    y_res = run_multi_agent(
        "wmsr",
        X0,
        y0,
        grid,
        y_actual,
        N=N,
        T=T,
        meeting_every=meeting_every,
        F=F,
        n_faulty=n_faulty,
        faulty_set=faulty_set,
        faulty_bias=faulty_bias,
        faulty_noise_std=faulty_noise_std,
        clean_noise_std=clean_noise_std,
        seed=seed,
        freeze_after_meeting=freeze_after_meeting,
        fuse_predictions=fuse_predictions,
        verbose=True,
    )
    Z_res = y_res.reshape(GRID_RES, GRID_RES)

    # Print numeric proof
    print(f"[metric] RMSE init    : {rmse(y_init, y_actual):.4f}")
    print(f"[metric] RMSE nonres  : {rmse(y_nonres, y_actual):.4f}")
    print(f"[metric] RMSE resilient: {rmse(y_res, y_actual):.4f}")

    # 3D 2x2 plot
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
    out = BASE_DIR / "paper_style_3d_4panel.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[main] saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
