import itertools
from typing import List, Dict

class CandidateGenerator:
    def __init__(self):
        pass
        
    def generate(self, feature_names: List[str]) -> List[Dict]:
        candidates = []
        # Unary numerical transformations
        for f in feature_names:
            candidates.append({"id": f"LOG({f})", "transform": "LOG", "features": [f]})
            candidates.append({"id": f"SQRT({f})", "transform": "SQRT", "features": [f]})
            candidates.append({"id": f"SQUARE({f})", "transform": "SQUARE", "features": [f]})
            candidates.append({"id": f"ABS({f})", "transform": "ABS", "features": [f]})
            
        # Binary numerical transformations (limit to first 100 pairs for speed if needed, but we keep simple)
        for f1, f2 in itertools.combinations(feature_names[:20], 2): # limit to 20 features to avoid combinatorics explosion
            candidates.append({"id": f"ADD({f1},{f2})", "transform": "ADD", "features": [f1, f2]})
            candidates.append({"id": f"SUB({f1},{f2})", "transform": "SUB", "features": [f1, f2]})
            candidates.append({"id": f"MUL({f1},{f2})", "transform": "MUL", "features": [f1, f2]})
            candidates.append({"id": f"DIV({f1},{f2})", "transform": "DIV", "features": [f1, f2]})
            
        return candidates
