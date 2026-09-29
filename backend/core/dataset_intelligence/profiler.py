import pandas as pd
import numpy as np
import json
from typing import Dict, Any

class DatasetProfiler:
    def __init__(self, dataset_name: str, task_type: str = 'classification'):
        self.dataset_name = dataset_name
        self.task_type = task_type
        # Explicit numeric vector ordering
        self.vector_keys = [
            "n_rows", "n_cols", "n_numeric", "n_categorical", "n_boolean", "n_datetime",
            "missing_ratio", "cols_with_missing", "mean_missingness_per_feature", "max_feature_missingness",
            "mean_cardinality", "max_cardinality", "avg_abs_correlation", "max_abs_correlation"
        ]

    def _detect_types(self, df: pd.DataFrame, target_col: str):
        import pandas.api.types as ptypes
        types = {}
        for col in df.columns:
            if col == target_col:
                continue
            if ptypes.is_bool_dtype(df[col]):
                types[col] = 'boolean'
            elif ptypes.is_datetime64_any_dtype(df[col]):
                types[col] = 'datetime'
            elif ptypes.is_numeric_dtype(df[col]):
                types[col] = 'numerical'
            else:
                types[col] = 'categorical'
        return types

    def profile(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        feature_types = self._detect_types(df, target_col)
        n_rows, n_cols = X.shape
        
        numeric_cols = [c for c, t in feature_types.items() if t == 'numerical']
        cat_cols = [c for c, t in feature_types.items() if t == 'categorical']
        bool_cols = [c for c, t in feature_types.items() if t == 'boolean']
        dt_cols = [c for c, t in feature_types.items() if t == 'datetime']
        
        # Missingness
        total_missing = int(X.isna().sum().sum())
        missing_ratio = float(total_missing / (n_rows * n_cols)) if n_rows * n_cols > 0 else 0.0
        cols_missing = int((X.isna().sum() > 0).sum())
        mean_miss_per_feat = float(X.isna().mean().mean()) if n_cols > 0 else 0.0
        max_miss_feat = float(X.isna().mean().max()) if n_cols > 0 else 0.0
        
        # Cardinality
        if cat_cols:
            cardinalities = X[cat_cols].nunique()
            mean_card = float(cardinalities.mean())
            max_card = int(cardinalities.max())
        else:
            mean_card = 0.0
            max_card = 0
            
        # Numerical stats (Dataset level)
        if numeric_cols:
            corr_mat = X[numeric_cols].corr().abs().to_numpy(copy=True)
            np.fill_diagonal(corr_mat, np.nan)
            avg_abs_corr = float(np.nanmean(corr_mat)) if not np.isnan(corr_mat).all() else 0.0
            max_abs_corr = float(np.nanmax(corr_mat)) if not np.isnan(corr_mat).all() else 0.0
        else:
            avg_abs_corr = 0.0
            max_abs_corr = 0.0
            
        # Target profile
        target_profile = {}
        if self.task_type == 'classification':
            counts = y.value_counts(normalize=True)
            target_profile = {
                'n_classes': len(counts),
                'majority_class_proportion': float(counts.max()),
                'minority_class_proportion': float(counts.min()),
                'class_imbalance_ratio': float(counts.max() / counts.min()) if counts.min() > 0 else 0.0
            }
        else:
            target_profile = {
                'target_mean': float(y.mean()),
                'target_std': float(y.std()),
                'target_min': float(y.min()),
                'target_max': float(y.max()),
                'target_skewness': float(y.skew()) if n_rows > 2 else 0.0
            }
            
        # Feature profiles
        feature_profiles = {}
        for col in X.columns:
            ftype = feature_types[col]
            missing_val = float(X[col].isna().mean())
            unique_c = int(X[col].nunique())
            fprof = {
                'feature_name': col,
                'feature_type': ftype,
                'missing_ratio': missing_val,
                'unique_count': unique_c,
                'unique_ratio': float(unique_c / n_rows) if n_rows > 0 else 0.0,
                'mean': None, 'std': None, 'min': None, 'max': None, 'median': None, 'skewness': None, 'cardinality': None
            }
            
            if ftype == 'numerical':
                fprof.update({
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std()),
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'median': float(X[col].median()),
                    'skewness': float(X[col].skew()) if n_rows > 2 else 0.0
                })
            elif ftype == 'categorical':
                fprof['cardinality'] = unique_c
                
            feature_profiles[col] = fprof
            
        def clean_nans(d):
            if isinstance(d, dict):
                return {k: clean_nans(v) for k, v in d.items()}
            elif isinstance(d, float) and (np.isnan(d) or np.isinf(d)):
                return None
            return d
            
        structural = {
            'n_rows': int(n_rows),
            'n_cols': int(n_cols),
            'n_numeric': len(numeric_cols),
            'n_categorical': len(cat_cols),
            'n_boolean': len(bool_cols),
            'n_datetime': len(dt_cols),
            'task_type': self.task_type
        }
        
        raw_context = {
            'dataset_name': self.dataset_name,
            'profiler_version': '1.0',
            'structural': structural,
            'missingness': {
                'total_missing': total_missing,
                'missing_ratio': missing_ratio,
                'cols_with_missing': cols_missing,
                'mean_missingness_per_feature': mean_miss_per_feat,
                'max_feature_missingness': max_miss_feat
            },
            'categorical': {
                'mean_cardinality': mean_card,
                'max_cardinality': max_card
            },
            'numerical': {
                'avg_abs_correlation': avg_abs_corr,
                'max_abs_correlation': max_abs_corr
            },
            'target': target_profile,
            'feature_profiles': feature_profiles
        }
        
        raw_context = clean_nans(raw_context)
        
        vector = []
        for key in self.vector_keys:
            val = None
            if key in raw_context['structural']: val = raw_context['structural'][key]
            elif key in raw_context['missingness']: val = raw_context['missingness'][key]
            elif key in raw_context['categorical']: val = raw_context['categorical'][key]
            elif key in raw_context['numerical']: val = raw_context['numerical'][key]
            vector.append(val if val is not None else 0.0)
            
        return {
            'raw_context': raw_context,
            'numeric_context_vector': vector,
            'vector_keys': self.vector_keys
        }

    def serialize(self, profile: Dict[str, Any], filepath: str):
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=4)
            
    def deserialize(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            return json.load(f)
