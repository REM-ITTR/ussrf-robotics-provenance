import sys
from pathlib import Path
import json
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.hashing import sha256_bytes
from core.manifest import write_manifest

IN_PATH = REPO_ROOT / "datasets" / "sample_numpy" / "telemetry_small.npz"
OUT_DIR = REPO_ROOT / "outputs" / "mvp3"
PROOF_DIR = REPO_ROOT / "proofs" / "mvp3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROOF_DIR.mkdir(parents=True, exist_ok=True)

tele = np.load(IN_PATH)["x"]
orig_hash = sha256_bytes(tele.tobytes())

# Deterministic reduction: drop consecutive duplicate rows
kept = [0]
for i in range(1, tele.shape[0]):
    if not np.array_equal(tele[i], tele[i-1]):
        kept.append(i)

reduced = tele[kept]
reduced_hash = sha256_bytes(reduced.tobytes())

# Expansion map = run-length encoding implied by kept indices
# (We preserve exact original by storing kept indices + original length)
manifest = {
    "mvp": "mvp3_telemetry_reduction_numpy",
    "input": str(IN_PATH.relative_to(REPO_ROOT)),
    "original_shape": list(tele.shape),
    "reduced_shape": list(reduced.shape),
    "rr_rows": float(reduced.shape[0] / tele.shape[0]),
    "original_sha256": orig_hash,
    "reduced_sha256": reduced_hash,
    "expansion_map": {
        "kept_indices": kept,
        "original_len": int(tele.shape[0]),
        "note": "expand by repeating reduced rows across spans between kept indices",
    },
}

np.savez(OUT_DIR / "reduced_telemetry.npz", x=reduced)
write_manifest(str(PROOF_DIR / "manifest.json"), manifest)

# "Policy fingerprint" demo: treat any bytes as a policy blob
policy_blob = b"demo_policy_checkpoint_v1"
policy_fp = sha256_bytes(policy_blob)
(PROOF_DIR / "policy_fingerprint.txt").write_text(policy_fp + "\n", encoding="utf-8")

metrics = {
    "original_rows": int(tele.shape[0]),
    "reduced_rows": int(reduced.shape[0]),
    "rr_rows": float(reduced.shape[0] / tele.shape[0]),
    "bytes_original": int(tele.nbytes),
    "bytes_reduced": int(reduced.nbytes),
    "bytes_saved": int(tele.nbytes - reduced.nbytes),
}

with open(PROOF_DIR / "telemetry_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, sort_keys=True)

(PROOF_DIR / "verification_report.md").write_text(
    "# MVP3 Verification Report\n\n"
    "- Deterministic consecutive-duplicate telemetry reduction\n"
    "- Policy identity fingerprint (demo blob)\n\n"
    f"- RR (rows): {metrics['rr_rows']:.6f}\n"
    f"- Bytes saved: {metrics['bytes_saved']}\n"
    f"- Original SHA256: `{orig_hash}`\n"
    f"- Reduced SHA256: `{reduced_hash}`\n"
    f"- Policy fingerprint: `{policy_fp}`\n",
    encoding="utf-8"
)

print("OK: wrote proofs/mvp3/{manifest.json, policy_fingerprint.txt, telemetry_metrics.json, verification_report.md}")
