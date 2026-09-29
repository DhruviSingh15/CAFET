import numpy as np
import pandas as pd
from scipy.stats import skew, pearsonr, spearmanr
import warnings

def _safe_float(val):
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except (ValueError, TypeError):
        return 0.0

def compute_source_feature_stats(series: pd.Series) -> np.ndarray:
    """Returns 15-dim array of single-feature stats."""
    if series is None or series.empty:
        return np.zeros(15)
        
    n = len(series)
    missing = series.isna().sum()
    valid = series.dropna()
    valid_n = len(valid)
    
    if valid_n == 0:
        return np.zeros(15)
        
    # numeric check
    if not pd.api.types.is_numeric_dtype(valid):
        # convert to categories/numeric if possible, else return safe categorical stats
        try:
            valid = pd.to_numeric(valid, errors='coerce').dropna()
            valid_n = len(valid)
        except:
            pass
            
    unique_count = valid.nunique()
    unique_ratio = unique_count / valid_n if valid_n > 0 else 0
    missing_ratio = missing / n
    
    if valid_n == 0:
        return np.array([missing_ratio, unique_count, unique_ratio] + [0.0]*12)
        
    mean = valid.mean()
    std = valid.std()
    min_val = valid.min()
    max_val = valid.max()
    median = valid.median()
    
    # Ignore warnings for skew and other stats
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skewness = skew(valid) if valid_n > 2 else 0.0
        
    zero_ratio = (valid == 0).sum() / valid_n
    finite_ratio = np.isfinite(valid).sum() / valid_n
    positive_ratio = (valid > 0).sum() / valid_n
    negative_ratio = (valid < 0).sum() / valid_n
    abs_mean = np.abs(valid).mean()
    iqr = valid.quantile(0.75) - valid.quantile(0.25)
    
    stats = [
        missing_ratio,
        unique_count,
        unique_ratio,
        mean,
        std,
        min_val,
        max_val,
        median,
        skewness,
        zero_ratio,
        finite_ratio,
        positive_ratio,
        negative_ratio,
        abs_mean,
        iqr
    ]
    return np.array([_safe_float(x) for x in stats])

def compute_pairwise_stats(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    """Returns 12-dim array of pairwise stats."""
    if s1 is None or s2 is None or s1.empty or s2.empty:
        return np.zeros(12)
        
    df = pd.DataFrame({'s1': pd.to_numeric(s1, errors='coerce'), 's2': pd.to_numeric(s2, errors='coerce')})
    n = len(df)
    
    # Overlap and joint missingness
    s1_valid = ~df['s1'].isna()
    s2_valid = ~df['s2'].isna()
    overlap = (s1_valid & s2_valid).sum() / n
    joint_missing = (~s1_valid & ~s2_valid).sum() / n
    
    df_valid = df.dropna()
    valid_n = len(df_valid)
    
    if valid_n < 2:
        return np.array([0, 0, 0, 0, 0, 0, 0, overlap, joint_missing, 0, 0, 0])
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson, _ = pearsonr(df_valid['s1'], df_valid['s2'])
        spearman, _ = spearmanr(df_valid['s1'], df_valid['s2'])
        cov = df_valid['s1'].cov(df_valid['s2'])
        
    abs_corr = np.abs(pearson) if not np.isnan(pearson) else 0.0
    
    std1 = df_valid['s1'].std()
    std2 = df_valid['s2'].std()
    scale_ratio = std1 / std2 if std2 != 0 else 0.0
    std_ratio = scale_ratio
    
    mean1 = df_valid['s1'].mean()
    mean2 = df_valid['s2'].mean()
    mean_ratio = mean1 / mean2 if mean2 != 0 else 0.0
    
    zero_co_occurrence = ((df_valid['s1'] == 0) & (df_valid['s2'] == 0)).sum() / valid_n
    
    sign1 = np.sign(df_valid['s1'])
    sign2 = np.sign(df_valid['s2'])
    sign_agreement = (sign1 == sign2).sum() / valid_n
    
    rank_corr = spearman
    
    stats = [
        pearson,
        spearman,
        cov,
        abs_corr,
        scale_ratio,
        mean_ratio,
        std_ratio,
        overlap,
        joint_missing,
        zero_co_occurrence,
        sign_agreement,
        rank_corr
    ]
    return np.array([_safe_float(x) for x in stats])

def apply_transformation(transform: str, s1: pd.Series, s2: pd.Series = None) -> pd.Series:
    s1 = pd.to_numeric(s1, errors='coerce')
    if s2 is not None:
        s2 = pd.to_numeric(s2, errors='coerce')
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if transform == "LOG":
            # shift to avoid log(0)
            shift = 1.0 if s1.min() >= 0 else -s1.min() + 1.0
            return np.log(s1 + shift)
        elif transform == "SQRT":
            # shift to avoid sqrt(<0)
            shift = 0.0 if s1.min() >= 0 else -s1.min()
            return np.sqrt(s1 + shift)
        elif transform == "SQUARE":
            return np.square(s1)
        elif transform == "ABS":
            return np.abs(s1)
        elif transform == "ADD" and s2 is not None:
            return s1 + s2
        elif transform == "SUB" and s2 is not None:
            return s1 - s2
        elif transform == "MUL" and s2 is not None:
            return s1 * s2
        elif transform == "DIV" and s2 is not None:
            return s1 / s2.replace(0, np.nan)
            
    return pd.Series(np.nan, index=s1.index)

def compute_output_stats(out_series: pd.Series) -> np.ndarray:
    """Returns 12-dim array of transformation-output stats."""
    if out_series is None or out_series.empty:
        return np.zeros(12)
        
    n = len(out_series)
    missing = out_series.isna().sum()
    missing_ratio = missing / n
    
    valid = out_series.dropna()
    valid_n = len(valid)
    
    if valid_n == 0:
        return np.array([missing_ratio] + [0.0]*11)
        
    inf_count = np.isinf(valid).sum()
    inf_ratio = inf_count / n
    
    # filter out infs for stats
    valid = valid[np.isfinite(valid)]
    valid_n = len(valid)
    finite_ratio = valid_n / n
    
    if valid_n == 0:
        return np.array([missing_ratio, finite_ratio, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf_ratio])
        
    unique_count = valid.nunique()
    unique_ratio = unique_count / valid_n
    variance = valid.var()
    std = valid.std()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skewness = skew(valid) if valid_n > 2 else 0.0
        
    zero_ratio = (valid == 0).sum() / valid_n
    pos_ratio = (valid > 0).sum() / valid_n
    neg_ratio = (valid < 0).sum() / valid_n
    range_val = valid.max() - valid.min()
    iqr = valid.quantile(0.75) - valid.quantile(0.25)
    
    stats = [
        missing_ratio,
        finite_ratio,
        unique_ratio,
        variance,
        std,
        skewness,
        zero_ratio,
        pos_ratio,
        neg_ratio,
        range_val,
        iqr,
        inf_ratio
    ]
    return np.array([_safe_float(x) for x in stats])

def build_micro_context(df: pd.DataFrame, transform: str, source_features: list) -> np.ndarray:
    """
    Returns 54-dim vector:
    15 (source1) + 15 (source2) + 12 (pairwise) + 12 (output)
    """
    if not source_features or source_features[0] not in df.columns:
        return np.zeros(54)
        
    s1 = df[source_features[0]]
    s1_stats = compute_source_feature_stats(s1)
    
    if len(source_features) > 1 and source_features[1] in df.columns:
        s2 = df[source_features[1]]
        s2_stats = compute_source_feature_stats(s2)
        pair_stats = compute_pairwise_stats(s1, s2)
        out_s = apply_transformation(transform, s1, s2)
    else:
        s2_stats = np.zeros(15)
        pair_stats = np.zeros(12)
        out_s = apply_transformation(transform, s1, None)
        
    out_stats = compute_output_stats(out_s)
    
    return np.concatenate([s1_stats, s2_stats, pair_stats, out_stats])

MICRO_CONTEXT_DIM = 54
