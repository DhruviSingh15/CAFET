import numpy as np
import pandas as pd
from typing import Dict

class CandidateEvaluator:
    def apply_transform(self, X: pd.DataFrame, candidate: Dict) -> pd.Series:
        transform = candidate['transform']
        features = candidate['features']
        
        try:
            if transform == 'LOG':
                val = X[features[0]]
                return np.log(np.clip(val, a_min=1e-5, a_max=None))
            elif transform == 'SQRT':
                val = X[features[0]]
                return np.sqrt(np.clip(val, a_min=0, a_max=None))
            elif transform == 'SQUARE':
                return np.square(X[features[0]])
            elif transform == 'ABS':
                return np.abs(X[features[0]])
            elif transform == 'ADD':
                return X[features[0]] + X[features[1]]
            elif transform == 'SUB':
                return X[features[0]] - X[features[1]]
            elif transform == 'MUL':
                return X[features[0]] * X[features[1]]
            elif transform == 'DIV':
                denom = X[features[1]]
                denom = np.where(denom == 0, 1e-5, denom)
                return X[features[0]] / denom
            else:
                raise ValueError(f"Unknown transform {transform}")
        except Exception as e:
            raise ValueError(f"Failed to apply {transform}: {str(e)}")
