import numpy as np
from typing import Dict, Any, Tuple

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def greedy_cluster(vectors: np.ndarray, threshold: float = 0.95) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Simple greedy clustering:
    - pick next unassigned as a representative
    - assign any vector with cosine similarity >= threshold to that cluster

    Returns:
      reps_idx: indices of representative vectors
      assignment: map idx -> rep_idx
    """
    n = vectors.shape[0]
    assigned = np.full(n, False)
    reps = []
    assignment = {}

    for i in range(n):
        if assigned[i]:
            continue
        reps.append(i)
        assigned[i] = True
        assignment[i] = i

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if cosine_sim(vectors[i], vectors[j]) >= threshold:
                assigned[j] = True
                assignment[j] = i

    return np.array(reps, dtype=int), assignment

def summarize_clusters(assignment: Dict[int, int]) -> Dict[int, Any]:
    clusters: Dict[int, Any] = {}
    for idx, rep in assignment.items():
        clusters.setdefault(rep, {"rep": rep, "members": []})
        clusters[rep]["members"].append(idx)
    # sort members for stable output
    for rep in clusters:
        clusters[rep]["members"] = sorted(clusters[rep]["members"])
    return clusters
