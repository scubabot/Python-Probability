# multi_agent_resilience.py
from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel


# ----------------------------
# GP helpers
# ----------------------------
def build_gp(amp=1.0, ls=1.0, noi=1.0, *, optimize=True, seed=0):
    """
    Matérn GP with White noise.
    - optimize=True: learn hypers (sklearn optimizer on)
    - optimize=False: keep hypers fixed (optimizer=None)
    """
    kernel = C(amp) * Matern(length_scale=ls, nu=1.5) + WhiteKernel(noise_level=noi)
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5 if optimize else 0,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        random_state=seed,
    )


def extract_hypers(gp):
    """Extract (amplitude, length_scale, noise) from a fitted sklearn GP."""
    k = gp.kernel_
    amp = float(k.k1.k1.constant_value)   # ConstantKernel value
    ls  = float(k.k1.k2.length_scale)     # Matern length_scale
    noi = float(k.k2.noise_level)         # WhiteKernel noise_level
    return amp, ls, noi


# ----------------------------
# Resilience aggregators
# ----------------------------
def wmsr(vals, F):
    """
    W-MSR trimmed mean:
    remove F smallest + F largest, then average the rest.
    """
    vals = sorted([float(v) for v in vals])
    if len(vals) == 0:
        raise ValueError("wmsr got empty list.")
    if 2 * F >= len(vals) - 1:
        return float(np.mean(vals))
    return float(np.mean(vals[F:len(vals) - F]))


def aggregate(hypers_list, mode="mean", F=1):
    """
    hypers_list: list of (amp, ls, noi) across agents at a meeting.
    mode: 'mean' or 'wmsr'
    """
    amps = [h[0] for h in hypers_list]
    lss  = [h[1] for h in hypers_list]
    nois = [h[2] for h in hypers_list]

    if mode == "mean":
        return float(np.mean(amps)), float(np.mean(lss)), float(np.mean(nois))
    elif mode == "wmsr":
        return wmsr(amps, F), wmsr(lss, F), wmsr(nois, F)
    else:
        raise ValueError("mode must be 'mean' or 'wmsr'")


# ----------------------------
# Multi-agent run
# ----------------------------
def run_multi_agent(mode, X0, y0, grid, y_actual, *,
                    N=5, T=40, meeting_every=10, F=1,
                    noise_std=0.0, seed=0):
    """
    mode:
      - 'mean'  -> non-resilient hyperparameter aggregation
      - 'wmsr'  -> resilient (W-MSR) hyperparameter aggregation

    Returns:
      y_mean_avg: average posterior mean across agents at the end (on 'grid')
    """
    rng = np.random.default_rng(seed)
    tree = cKDTree(grid)

    # init agents
    agents = []
    for i in range(N):
        X = X0.copy()
        y = y0.copy()

        # used mask to avoid selecting same grid points again
        used = np.zeros(grid.shape[0], dtype=bool)
        _, idx0 = tree.query(X)
        used[idx0] = True

        gp = build_gp(optimize=True, seed=seed + i)
        agents.append({"X": X, "y": y, "used": used, "gp": gp})

    # iterations
    for t in range(1, T + 1):
        # local updates + new sample
        for a in agents:
            a["gp"].fit(a["X"], a["y"])
            y_mean, y_std = a["gp"].predict(grid, return_std=True)

            y_std_masked = y_std.copy()
            y_std_masked[a["used"]] = -np.inf
            idx_next = int(np.argmax(y_std_masked))

            x_new = grid[idx_next].reshape(1, 2)
            y_new = float(y_actual[idx_next] + rng.normal(0, noise_std))

            a["X"] = np.vstack([a["X"], x_new])
            a["y"] = np.append(a["y"], y_new)
            a["used"][idx_next] = True

        # meeting: aggregate hypers and push back
        if t % meeting_every == 0:
            hypers = []
            for a in agents:
                a["gp"].fit(a["X"], a["y"])
                hypers.append(extract_hypers(a["gp"]))

            agg_amp, agg_ls, agg_noi = aggregate(hypers, mode=mode, F=F)
            print(f"[{mode}] meeting @t={t:02d}: amp={agg_amp:.3f}, ls={agg_ls:.3f}, noi={agg_noi:.3f}")

            # rebuild all agents with fixed aggregated hypers
            for i, a in enumerate(agents):
                a["gp"] = build_gp(amp=agg_amp, ls=agg_ls, noi=agg_noi, optimize=False, seed=seed + i)

    # final: average predicted mean across agents
    preds = []
    for a in agents:
        a["gp"].fit(a["X"], a["y"])
        preds.append(a["gp"].predict(grid))

    y_mean_avg = np.mean(np.vstack(preds), axis=0)
    return y_mean_avg


# ----------------------------
# 3D plotting (paper-like)
# ----------------------------
def plot_3d_panel(ax, Z, title, elev=25, azim=-135):
    n = Z.shape[0]
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_title(title, pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def main():
    # Load MATLAB workspace
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

    # (b) Initial GP from starting points only
    gp_init = build_gp(optimize=True, seed=123)
    gp_init.fit(X0, y0)
    y_init = gp_init.predict(grid)
    Z_init = y_init.reshape(GRID_RES, GRID_RES)

    # Multi-agent settings (start simple)
    N = 5
    T = 40
    meeting_every = 10
    F = 1
    noise_std = 0.0

    # (c) Non-resilient: mean hypers
    y_nonres = run_multi_agent(
        "mean", X0, y0, grid, y_actual,
        N=N, T=T, meeting_every=meeting_every, F=F,
        noise_std=noise_std, seed=0
    )
    Z_nonres = y_nonres.reshape(GRID_RES, GRID_RES)

    # (d) Resilient: W-MSR hypers
    y_res = run_multi_agent(
        "wmsr", X0, y0, grid, y_actual,
        N=N, T=T, meeting_every=meeting_every, F=F,
        noise_std=noise_std, seed=0
    )
    Z_res = y_res.reshape(GRID_RES, GRID_RES)

    # 3D 2x2 figure (paper-style)
    fig = plt.figure(figsize=(14, 9))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_3d_panel(ax1, Z_actual, "(a) Actual environmental GP")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_3d_panel(ax2, Z_init, "(b) Initial knowledge of the GP")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    plot_3d_panel(ax3, Z_nonres, "(c) Learned GP — non-resilient (mean hypers)")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    plot_3d_panel(ax4, Z_res, "(d) Learned GP — resilient (W-MSR hypers)")

    plt.tight_layout()
    plt.savefig("paper_style_3d_4panel.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
