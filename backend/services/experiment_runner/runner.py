import time
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from backend.core.preprocessing.preprocessor import BaselinePreprocessor
from backend.core.candidate_generation.generator import CandidateGenerator
from backend.core.candidate_evaluation.evaluator import CandidateEvaluator
from backend.models.baseline_model.model import BaselineModel

class ExperimentRunner:
    def __init__(self, dataset_name, df, target_col, task='classification', seed=42):
        self.dataset_name = dataset_name
        self.df = df
        self.target_col = target_col
        self.task = task
        self.seed = seed
        self.results = {}
        
    def run(self):
        start_time = time.time()
        
        # 6. Data Splitting: 70% Train, 15% Val, 15% Test
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=self.seed)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.17647, random_state=self.seed)
        
        # Preprocessing
        preprocessor = BaselinePreprocessor()
        X_train_prep = preprocessor.fit_transform(X_train)
        X_val_prep = preprocessor.transform(X_val)
        X_test_prep = preprocessor.transform(X_test)
        
        # Baseline model
        baseline_model = BaselineModel(task=self.task, random_state=self.seed)
        t0 = time.time()
        baseline_model.fit(X_train_prep, y_train)
        training_time = time.time() - t0
        
        t0 = time.time()
        baseline_val_score = baseline_model.evaluate(X_val_prep, y_val)
        evaluation_time = time.time() - t0
        
        baseline_test_score = baseline_model.evaluate(X_test_prep, y_test)
        
        self.results['baseline'] = {
            'baseline_val_score': baseline_val_score,
            'baseline_test_score': baseline_test_score,
            'training_time': training_time,
            'evaluation_time': evaluation_time
        }
        
        # Candidate Generation & Evaluation
        generator = CandidateGenerator()
        candidates = generator.generate(X_train_prep.columns.tolist())
        evaluator = CandidateEvaluator()
        
        candidate_results = []
        successful_evals = 0
        failed_evals = 0
        
        for cand in candidates:
            try:
                t_c0 = time.time()
                train_feat = evaluator.apply_transform(X_train_prep, cand)
                val_feat = evaluator.apply_transform(X_val_prep, cand)
                
                if train_feat.isna().any() or np.isinf(train_feat).any():
                    raise ValueError("NaN or Inf generated")
                
                X_train_cand = X_train_prep.copy()
                X_val_cand = X_val_prep.copy()
                X_train_cand[cand['id']] = train_feat
                X_val_cand[cand['id']] = val_feat
                
                model = BaselineModel(task=self.task, random_state=self.seed)
                model.fit(X_train_cand, y_train)
                val_score = model.evaluate(X_val_cand, y_val)
                
                eval_time = time.time() - t_c0
                
                candidate_results.append({
                    'dataset': self.dataset_name,
                    'candidate_id': cand['id'],
                    'transformation': cand['transform'],
                    'source_features': cand['features'],
                    'validation_score': val_score,
                    'gain': val_score - baseline_val_score,
                    'evaluation_time': eval_time,
                    'status': 'success',
                    'error': None
                })
                successful_evals += 1
            except Exception as e:
                candidate_results.append({
                    'dataset': self.dataset_name,
                    'candidate_id': cand['id'],
                    'transformation': cand['transform'],
                    'source_features': cand['features'],
                    'validation_score': None,
                    'gain': None,
                    'evaluation_time': time.time() - t_c0,
                    'status': 'failed',
                    'error': str(e)
                })
                failed_evals += 1
                
        # Feature Selection (Rank by validation score)
        success_cands = [c for c in candidate_results if c['status'] == 'success']
        success_cands.sort(key=lambda x: x['validation_score'], reverse=True)
        
        best_candidate = None
        best_cand_test_score = baseline_test_score
        
        if success_cands:
            best_candidate = success_cands[0]
            cand = next(c for c in candidates if c['id'] == best_candidate['candidate_id'])
            
            train_feat = evaluator.apply_transform(X_train_prep, cand)
            test_feat = evaluator.apply_transform(X_test_prep, cand)
            
            X_train_cand = X_train_prep.copy()
            X_test_cand = X_test_prep.copy()
            X_train_cand[cand['id']] = train_feat
            X_test_cand[cand['id']] = test_feat
            
            final_model = BaselineModel(task=self.task, random_state=self.seed)
            final_model.fit(X_train_cand, y_train)
            best_cand_test_score = final_model.evaluate(X_test_cand, y_test)
            
        runtime = time.time() - start_time
        
        self.results['summary'] = {
            'dataset': self.dataset_name,
            'task': self.task,
            'rows': len(self.df),
            'features': len(X.columns),
            'baseline_val_score': baseline_val_score,
            'best_candidate_val_score': best_candidate['validation_score'] if best_candidate else None,
            'best_candidate': best_candidate['candidate_id'] if best_candidate else None,
            'gain': best_candidate['gain'] if best_candidate else 0,
            'candidate_count': len(candidates),
            'successful_evaluations': successful_evals,
            'failed_evaluations': failed_evals,
            'runtime': runtime,
            'baseline_test_score': baseline_test_score,
            'best_candidate_test_score': best_cand_test_score
        }
        self.results['candidates'] = candidate_results
        return self.results
