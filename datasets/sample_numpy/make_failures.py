import numpy as np
from pathlib import Path

Path("datasets/sample_numpy").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)

# episodes: each is a 16-d vector "terminal state signature"
# create 4 failure "modes" with noise + duplicates
modes = rng.normal(size=(4, 16)).astype(np.float32)
failures = []
labels = []

for mode_id in range(4):
    for _ in range(25):
        v = modes[mode_id] + rng.normal(scale=0.05, size=(16,)).astype(np.float32)
        failures.append(v)
        labels.append(1)  # failure

# add duplicates
failures.extend(failures[:10])
labels.extend([1]*10)

failures = np.vstack(failures).astype(np.float32)
labels = np.array(labels, dtype=np.int32)

np.savez("datasets/sample_numpy/failures_small.npz", x=failures, y=labels)
print("Wrote datasets/sample_numpy/failures_small.npz", failures.shape, labels.shape)
