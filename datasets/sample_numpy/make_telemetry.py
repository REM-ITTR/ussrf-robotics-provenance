import numpy as np
from pathlib import Path

Path("datasets/sample_numpy").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(21)

# telemetry: 500 timesteps, each with 10 float features
tele = rng.normal(size=(500, 10)).astype(np.float32)

# Inject repeated "steady" segments (duplicates-ish)
tele[100:150] = tele[100]
tele[300:330] = tele[300]

np.savez("datasets/sample_numpy/telemetry_small.npz", x=tele)
print("Wrote datasets/sample_numpy/telemetry_small.npz", tele.shape)
