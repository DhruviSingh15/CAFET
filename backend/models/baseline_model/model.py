from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

class BaselineModel:
    def __init__(self, task='classification', random_state=42):
        self.task = task
        if task == 'classification':
            self.model = LogisticRegression(random_state=random_state, max_iter=1000)
            self.metric = accuracy_score
        else:
            self.model = RandomForestRegressor(random_state=random_state, n_estimators=50)
            self.metric = r2_score
            
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def evaluate(self, X, y):
        preds = self.model.predict(X)
        return self.metric(y, preds)
