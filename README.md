# USSRF Robotics Provenance — Proof Artifacts (MVP 1–3)

This repository publishes **verifiable proof artifacts** demonstrating a
non-invasive **Reduce → Expand → Verify** infrastructure pattern applied to
robotics-adjacent data artifacts such as simulation trajectories, failure
archives, and telemetry streams.

The purpose is to prove **containment, provenance, and auditability** —
not to modify models or training pipelines.

---

## What this is

- Deterministic data reduction with explicit expansion maps
- Cryptographic fingerprints (SHA-256) proving integrity
- Clear separation of deterministic vs heuristic components
- Runnable examples that generate verification reports
- Artifact-level provenance suitable for reproducibility and audits

---

## What this is not

- Not a robotics control system
- Not an RL algorithm
- Not a simulator
- Not production-tuned profiles for any specific vendor stack

USSRF operates **only on data artifacts**, never on model internals.

---

## Proof Artifacts Overview

Each MVP produces **files you can inspect** — not claims.

The pattern is:
1. Generate sample data
2. Reduce it
3. Expand it
4. Prove integrity with hashes and manifests

---

## MVP1 — Deterministic Corpus Provenance  
**Reduce → Expand → Verify (lossless)**

### What this proves
- Reduction preserves full information
- Expanded data is **byte-identical** to original
- SHA-256 hashes match before and after
- Expansion map is explicit and auditable

### Run
```bash
python datasets/sample_numpy/make_sample.py
python examples/mvp1_corpus_provenance_numpy.py
