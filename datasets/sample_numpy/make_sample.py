import numpy as np
from pathlib import Path

Path("datasets/sample_numpy").mkdir(parents=True, exist_ok=True)

# Create a small "trajectory-like" array: rows = timesteps, cols = state dims
# Add duplicates intentionally so reduction is visible
rng = np.random.default_rng(7)
base = rng.integers(0, 10, size=(40, 6), dtype=np.int32)

# Inject exact duplicates
arr = np.vstack([base, base[:10], base[5:15], base[:5]])

np.savez("datasets/sample_numpy/traj_small.npz", arr=arr)
print("Wrote datasets/sample_numpy/traj_small.npz with shape:", arr.shape)
