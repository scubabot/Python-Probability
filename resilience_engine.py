from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
# 1) Loading MATLAB workspace
BASE_DIR = Path(__file__).parent
mat = scipy.io.loadmat(BASE_DIR / "matlab_workspace.mat")
g = mat["g"][0, 0]
# Extracting data from MATLAB struct
X_train = g["xi"]                    #from our dataset (43, 2)
y_train = g["yi"].reshape(-1)        # from our dataset(43,)
grid = g["grid"]                     # (625, 2)
y_actual = g["yactual"].reshape(-1)  # (625,)
print("Loaded data:")
print("  X_train:", X_train.shape)
print("  y_train:", y_train.shape)
print("  grid:", grid.shape)
print("  y_actual:", y_actual.shape)
# 2 Training Gaussian Process
kernel = (
    C(1.0, (1e-2, 1e3)) *
    Matern(length_scale=1.0, nu=1.5) +
    WhiteKernel(noise_level=1.0)
)
gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=5,
    random_state=42
)
print("\nTraining GP in Python...")
gp.fit(X_train, y_train)
print("Learned kernel:", gp.kernel_)
# 3) Predict on full grid
y_pred, y_std = gp.predict(grid, return_std=True)
# MATLAB grid is 25x25 (since 625 = 25 * 25)
GRID_RES = int(np.sqrt(grid.shape[0]))
Z_actual = y_actual.reshape(GRID_RES, GRID_RES)
Z_pred   = y_pred.reshape(GRID_RES, GRID_RES)
Z_std    = y_std.reshape(GRID_RES, GRID_RES)
# 4) Plot results
plt.figure(figsize=(14, 4))
# Actual environment
plt.subplot(1, 3, 1)
plt.title("Actual Environment (MATLAB yactual)")
plt.imshow(Z_actual, origin="lower")
plt.colorbar(label="Value")
# GP mean
plt.subplot(1, 3, 2)
plt.title("GP Mean (Python)")
plt.imshow(Z_pred, origin="lower")
plt.scatter(
    X_train[:, 0] * 0 + (GRID_RES // 2),  #just a placeholder for indexing can ne removed
    X_train[:, 1] * 0 + (GRID_RES // 2),
    s=5,
    color="white"
)
plt.colorbar(label="Mean")

#GP uncertainty
plt.subplot(1, 3, 3)
plt.title("GP Uncertainty (Std Dev)")
plt.imshow(Z_std, origin="lower")
plt.colorbar(label="Std")
plt.tight_layout()
plt.show()
