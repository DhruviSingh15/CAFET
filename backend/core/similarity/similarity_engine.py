import numpy as np
from typing import List, Dict, Any, Tuple

class Normalizer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, vectors: np.ndarray):
        if len(vectors) == 0:
            self.means = np.zeros(0)
            self.stds = np.ones(0)
            return
        
        self.means = np.nanmean(vectors, axis=0)
        self.stds = np.nanstd(vectors, axis=0)
        # Handle zero variance
        self.stds[self.stds == 0] = 1.0
        
        # Replace NaNs in mean/std if an entire column was NaN
        self.means = np.nan_to_num(self.means, nan=0.0)
        self.stds = np.nan_to_num(self.stds, nan=1.0)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise ValueError("Normalizer is not fitted yet.")
        v = np.copy(vectors)
        # NaN safe replacement before normalization 
        for i in range(v.shape[1]):
            nan_mask = np.isnan(v[:, i])
            v[nan_mask, i] = self.means[i]
        
        return (v - self.means) / self.stds

class SimilarityEngine:
    def __init__(self):
        pass

    def compute_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        # Euclidean distance
        dist = np.linalg.norm(vec_a - vec_b)
        return float(dist)

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        # 1 / (1 + distance)
        dist = self.compute_distance(vec_a, vec_b)
        return 1.0 / (1.0 + dist)
        
    def compute_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def rank(self, current_dataset_id: str, current_vector: List[float], 
             historical_contexts: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """
        historical_contexts: dict mapping dataset_id to numeric context vector.
        It should NOT contain the current_dataset_id.
        """
        if current_dataset_id in historical_contexts:
            # strictly remove it to enforce LODO
            historical_contexts = {k: v for k, v in historical_contexts.items() if k != current_dataset_id}

        if not historical_contexts:
            return []

        hist_ids = sorted(list(historical_contexts.keys()))
        hist_matrix = np.array([historical_contexts[hid] for hid in hist_ids], dtype=float)
        
        curr_vec = np.array(current_vector, dtype=float).reshape(1, -1)

        normalizer = Normalizer()
        normalizer.fit(hist_matrix) # Fit ONLY on historical available contexts

        norm_hist = normalizer.transform(hist_matrix)
        norm_curr = normalizer.transform(curr_vec)[0]

        results = []
        for i, hid in enumerate(hist_ids):
            sim = self.compute_similarity(norm_curr, norm_hist[i])
            dist = self.compute_distance(norm_curr, norm_hist[i])
            results.append({
                'dataset_id': hid,
                'dataset_name': hid,
                'similarity': sim,
                'distance': dist
            })

        # Deterministic sort: similarity DESC, dataset_id ASC
        results.sort(key=lambda x: (-x['similarity'], x['dataset_id']))
        
        for i, res in enumerate(results):
            res['rank'] = i + 1
            
        return results
        
    def similarity_matrix(self, contexts: Dict[str, List[float]]) -> Tuple[Dict[str, Dict[str, float]], Normalizer]:
        """Diagnostic matrix. Fits normalizer on ALL provided datasets."""
        ids = sorted(list(contexts.keys()))
        matrix = np.array([contexts[i] for i in ids], dtype=float)
        
        norm = Normalizer()
        norm.fit(matrix)
        norm_mat = norm.transform(matrix)
        
        res = {r: {} for r in ids}
        for i, r_id in enumerate(ids):
            for j, c_id in enumerate(ids):
                sim = self.compute_similarity(norm_mat[i], norm_mat[j])
                res[r_id][c_id] = sim
                
        return res, norm
