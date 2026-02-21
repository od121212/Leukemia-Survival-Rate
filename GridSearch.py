from sklearn.model_selection import GridSearchCV
import pandas as pd
from DataManagement import DataHandler, DefaultDataHandler
from ModelPipelines import DefaultPipeline
from sksurv.util import Surv
from config import PARAMS_RSF
from sksurv.metrics import concordance_index_censored



class ModelSelection:
    
    def __init__(self, model, param_grid, cv=5, n_jobs=-1):
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
        self.n_jobs = n_jobs
        self.grid = None
        self.best_model = None

    def cindex_scorer(self, estimator, X, y):
        pred = estimator.predict(X)
        return concordance_index_censored(
            y["OS_STATUS"],
            y["OS_YEARS"],
            pred
        )[0]
        
    def fit(self, X, y):
        self.grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.cindex_scorer,
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



if __name__ == "__main__":
    # Load data
    df = pd.read_csv("./X_train/clinical_train.csv", index_col=0)
    maf_df = pd.read_csv("./X_train/molecular_train.csv", index_col=0)
    target_df = pd.read_csv("./target_train.csv", index_col=0)

    # Build and fit pipeline
    data_handler = DefaultDataHandler(df, maf_df, target_df)
    pipeline_builder = DefaultPipeline(data_handler)
    pipeline = pipeline_builder.build_pipeline()

    y_surv = Surv.from_dataframe(
        event='OS_STATUS',   # 1 = event, 0 = censored
        time='OS_YEARS',
        data=data_handler.y
    )

    grid_search = ModelSelection(model=pipeline, param_grid=PARAMS_RSF, cv=5)
    grid_search.fit(data_handler.df, y_surv)
    print("Best Parameters:", grid_search.best_params())
    print("Best Score:", grid_search.best_score())

