import sys
from pathlib import Path
import json
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.hashing import sha256_bytes
from core.manifest import write_manifest
from core.failure_modes import greedy_cluster, summarize_clusters

IN_PATH = REPO_ROOT / "datasets" / "sample_numpy" / "failures_small.npz"
OUT_DIR = REPO_ROOT / "outputs" / "mvp2"
PROOF_DIR = REPO_ROOT / "proofs" / "mvp2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROOF_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(IN_PATH)
x = data["x"]  # vectors
y = data["y"]  # labels (1=failure)

fail = x[y == 1]
orig_hash = sha256_bytes(fail.tobytes())

# Heuristic step: cluster similar failures into unique modes
threshold = 0.95
reps_idx, assignment = greedy_cluster(fail, threshold=threshold)
clusters = summarize_clusters(assignment)

unique = fail[reps_idx]
reduced_hash = sha256_bytes(unique.tobytes())

# Deterministic expansion for analysis: any original failure maps to a rep
# (Not reconstructing exact original vectors; reconstructing "mode representative")
mode_map = {str(k): int(v) for k, v in assignment.items()}

np.savez(OUT_DIR / "unique_failure_modes.npz", x=unique)

with open(PROOF_DIR / "failure_modes.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "heuristic": True,
            "similarity": "cosine",
            "threshold": threshold,
            "total_failures": int(fail.shape[0]),
            "unique_modes": int(unique.shape[0]),
            "clusters": {str(rep): clusters[rep]["members"] for rep in clusters},
        },
        f,
        indent=2,
        sort_keys=True,
    )

manifest = {
    "mvp": "mvp2_failure_archive_modes_numpy",
    "input": str(IN_PATH.relative_to(REPO_ROOT)),
    "original_failures_shape": list(fail.shape),
    "unique_modes_shape": list(unique.shape),
    "rr_modes": float(unique.shape[0] / fail.shape[0]),
    "original_failures_sha256": orig_hash,
    "unique_modes_sha256": reduced_hash,
    "notes": {
        "deterministic_components": [
            "fingerprinting (sha256)",
            "artifact manifests",
            "stable saved representatives",
        ],
        "heuristic_components": [
            "failure clustering by cosine similarity threshold",
        ],
    },
}

write_manifest(str(PROOF_DIR / "manifest.json"), manifest)

report = []
report.append("# MVP2 Verification Report\n\n")
report.append("## What is deterministic vs heuristic?\n")
report.append("- Deterministic: fingerprints, manifests, saved representatives.\n")
report.append("- Heuristic: clustering failures into unique modes (cosine threshold).\n\n")
report.append(f"- Total failures: {fail.shape[0]}\n")
report.append(f"- Unique modes: {unique.shape[0]}\n")
report.append(f"- RR (modes): {manifest['rr_modes']:.6f}\n")
report.append(f"- Failures SHA256: `{orig_hash}`\n")
report.append(f"- Unique modes SHA256: `{reduced_hash}`\n")

(PROOF_DIR / "verification_report.md").write_text("".join(report), encoding="utf-8")

print("OK: wrote proofs/mvp2/{manifest.json, failure_modes.json, verification_report.md}")
