import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on PYTHONPATH so `import core...` works
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.hashing import sha256_bytes
from core.manifest import write_manifest

IN_PATH = REPO_ROOT / "datasets" / "sample_numpy" / "traj_small.npz"
OUT_DIR = REPO_ROOT / "outputs" / "mvp1"
PROOF_DIR = REPO_ROOT / "proofs" / "mvp1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PROOF_DIR.mkdir(parents=True, exist_ok=True)

arr = np.load(IN_PATH)["arr"]
orig_hash = sha256_bytes(arr.tobytes())

# Deterministic reduction: unique rows + inverse index (lossless expansion)
unique_rows, inverse = np.unique(arr, axis=0, return_inverse=True)
reduced_path = OUT_DIR / "reduced.npz"
np.savez(reduced_path, arr=unique_rows)
reduced_hash = sha256_bytes(unique_rows.tobytes())

# Deterministic expansion
expanded = unique_rows[inverse]
expanded_hash = sha256_bytes(expanded.tobytes())

byte_equal = (expanded.shape == arr.shape) and (expanded.tobytes() == arr.tobytes())
hash_equal = (expanded_hash == orig_hash)

manifest = {
    "mvp": "mvp1_corpus_provenance_numpy",
    "input": str(IN_PATH.relative_to(REPO_ROOT)),
    "output_reduced": str(reduced_path.relative_to(REPO_ROOT)),
    "original_shape": list(arr.shape),
    "reduced_shape": list(unique_rows.shape),
    "rr_rows": float(len(unique_rows) / len(arr)),
    "original_sha256": orig_hash,
    "reduced_sha256": reduced_hash,
    "expanded_sha256": expanded_hash,
    "verification": {
        "byte_equal_after_expand": bool(byte_equal),
        "sha256_equal_after_expand": bool(hash_equal),
    },
    "expansion_map": {
        "inverse_index_len": int(len(inverse)),
        "note": "expanded = reduced_unique[inverse_index]",
    },
}

write_manifest(str(PROOF_DIR / "manifest.json"), manifest)

(PROOF_DIR / "dataset_fingerprint.txt").write_text(orig_hash + "\n", encoding="utf-8")

report = []
report.append("# MVP1 Verification Report\n")
report.append(f"- Original shape: {tuple(arr.shape)}\n")
report.append(f"- Reduced shape: {tuple(unique_rows.shape)}\n")
report.append(f"- RR (rows): {manifest['rr_rows']:.6f}\n")
report.append(f"- Byte-equal after expand: {byte_equal}\n")
report.append(f"- SHA256 equal after expand: {hash_equal}\n")
report.append(f"- Original SHA256: `{orig_hash}`\n")

(PROOF_DIR / "verification_report.md").write_text("".join(report), encoding="utf-8")

print("OK: wrote proofs/mvp1/{manifest.json, dataset_fingerprint.txt, verification_report.md}")
