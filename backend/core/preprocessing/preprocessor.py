import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class BaselinePreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names_out_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        # Identify numeric and categorical columns
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.numeric_features),
                ('cat', cat_pipeline, self.categorical_features)
            ])
        
        self.preprocessor.fit(X)
        
        num_names = self.numeric_features
        try:
            cat_names = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(self.categorical_features).tolist()
        except:
            cat_names = []
            
        self.feature_names_out_ = num_names + cat_names
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted")
        
        X_trans = self.preprocessor.transform(X)
        return pd.DataFrame(X_trans, columns=self.feature_names_out_, index=X.index)
        
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
