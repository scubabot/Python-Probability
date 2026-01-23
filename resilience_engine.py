from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from scipy.spatial import cKDTree
from numpy.random import default_rng

# 1) Loading MATLAB workspace
BASE_DIR = Path(__file__).parent
mat = scipy.io.loadmat(BASE_DIR / "matlab_workspace.mat")
g = mat["g"][0, 0]

# Extracting data from MATLAB struct
X_train = np.array(g["xi"], dtype=float)                 # (43, 2)
y_train = np.array(g["yi"], dtype=float).reshape(-1)     # (43,)
grid = np.array(g["grid"], dtype=float)                  # (625, 2)
y_actual = np.array(g["yactual"], dtype=float).reshape(-1)  # (625,)

print("Loaded data:")
print("  X_train:", X_train.shape)
print("  y_train:", y_train.shape)
print("  grid:", grid.shape)
print("  y_actual:", y_actual.shape)

# 2) GP model
kernel = C(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=42)

rng = default_rng(0)
T = 30
noise_std = 0.0

tree = cKDTree(grid)
used = np.zeros(grid.shape[0], dtype=bool)
_, idx0 = tree.query(X_train)
used[idx0] = True

rmse_hist, unc_hist = [], []
amps, lss, nois = [], [], []

for t in range(T):
    gp.fit(X_train, y_train)

    # --- log hyperparameters AFTER fit ---
    k = gp.kernel_
    amps.append(float(k.k1.k1.constant_value))   # amplitude
    lss.append(float(k.k1.k2.length_scale))      # length-scale
    nois.append(float(k.k2.noise_level))         # noise

    # Predict on grid
    y_mean, y_std = gp.predict(grid, return_std=True)

    # Track progress
    rmse_hist.append(np.sqrt(np.mean((y_mean - y_actual) ** 2)))
    unc_hist.append(float(np.mean(y_std)))

    # Choose next sampling point = argmax uncertainty among UNUSED points
    y_std_masked = y_std.copy()
    y_std_masked[used] = -np.inf
    idx_next = int(np.argmax(y_std_masked))

    # Get new sample
    x_new = grid[idx_next].reshape(1, 2)
    y_new = float(y_actual[idx_next] + rng.normal(0, noise_std))

    # Append to dataset
    X_train = np.vstack([X_train, x_new])
    y_train = np.append(y_train, y_new)
    used[idx_next] = True

    print(f"iter {t+1:02d}: rmse={rmse_hist[-1]:.4f}, mean_std={unc_hist[-1]:.4f}, added idx={idx_next}")
    print("  hypers:", amps[-1], lss[-1], nois[-1])

# Final fit + prediction after loop
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(grid, return_std=True)

GRID_RES = int(np.sqrt(grid.shape[0]))
Z_actual = y_actual.reshape(GRID_RES, GRID_RES)
Z_pred   = y_pred.reshape(GRID_RES, GRID_RES)
Z_std    = y_std.reshape(GRID_RES, GRID_RES)

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.title("Actual Environment (MATLAB yactual)")
plt.imshow(Z_actual, origin="lower")
plt.colorbar(label="Value")

plt.subplot(1, 3, 2)
plt.title("GP Mean (Python, after loop)")
plt.imshow(Z_pred, origin="lower")
plt.colorbar(label="Mean")

plt.subplot(1, 3, 3)
plt.title("GP Uncertainty (Std Dev, after loop)")
plt.imshow(Z_std, origin="lower")
plt.colorbar(label="Std")

plt.tight_layout()
plt.savefig("results.png", dpi=300, bbox_inches="tight")
plt.show()
