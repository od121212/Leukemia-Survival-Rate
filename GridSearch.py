from sklearn.model_selection import GridSearchCV

class ModelSelection:
    
    def __init__(self, model, param_grid, cv=5, scoring=None, n_jobs=-1):
        """
        model: sklearn estimator
        param_grid: dict of hyperparameters
        cv: number of folds
        scoring: scoring metric
        n_jobs: number of parallel jobs (-1 = all cores)
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.grid = None
        self.best_model = None
        
    def fit(self, X, y):
        self.grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )
        
        self.grid.fit(X, y)
        self.best_model = self.grid.best_estimator_
        
        return self
    
    def best_params(self):
        return self.grid.best_params_
    
    def best_score(self):
        return self.grid.best_score_
    
    def predict(self, X):
        return self.best_model.predict(X)