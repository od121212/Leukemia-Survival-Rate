from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import make_column_selector
from abc import ABC, abstractmethod
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import logging
from DataManagement import DataHandler, DefaultDataHandler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Missing Values Transformer
class DropMissingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.cols_to_keep_ = None

    def fit(self, X, y=None):
        missing_ratio = X.isna().sum() / len(X)
        self.cols_to_keep_ = missing_ratio[missing_ratio <= self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X[self.cols_to_keep_]



# %%%%%%%%%%%%% Pipelines Builder %%%%%%%%%%%%%

# === Abstract Base Class for Pipelines ===
class ModelPipeline(ABC):

    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        pass
#------------------------------

#-------------- Pipeline Factory (Not Very usefull for now) ----------------
class PipelineFactory:

    @staticmethod
    def get_pipeline(model_type: str) -> ModelPipeline:
        if model_type == 'neural_network':
            pass
        elif model_type == 'linear_regression':
            pass
        else:
            return DefaultPipeline()  # Return a default pipeline for unknown types
#------------------------------
        


class DefaultPipeline(ModelPipeline):

    def build_pipeline(self) -> Pipeline:

        self.data_handler.decode_cytogen()
        self.data_handler.aggregator()
        self.data_handler.categorize()
        self.data_handler.drop_nan_target()

        X = self.data_handler.df
        y = self.data_handler.y

        # Float preprocessing
        float_pipe = Pipeline(steps=[
            ("impute_num", SimpleImputer(strategy="median"))
        ])

        # Categorial Preprocessing
        multi_cat_pipe = Pipeline(steps=[
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )),

            ("impute_cat", KNNImputer(
                n_neighbors=5,
                weights="distance"
            )),
        ])

        # binary categorical preprocessing
        binary_pipe = Pipeline(steps=[
            ("impute_cat", KNNImputer(
                n_neighbors=5,
                weights="distance"
            )),
        ])

        # columns transformer
        col_trans = ColumnTransformer(
            transformers=[
                ("num", float_pipe, self.data_handler.float_cols),
                ("cat", multi_cat_pipe, self.data_handler.categorical_cols),
                ("binary_cat", binary_pipe, self.data_handler.binary_cols)
            ],
            remainder='passthrough'
        )

        # === MODEL ===

        model = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )


        # return full Pipeline
        return Pipeline([
            ("drop_missing", DropMissingTransformer(threshold=0.2)),
            ('column_transformer', col_trans),
            ('model', model)
        ])
    



# %%%%%%%%%%%%% === MAIN TEST === %%%%%%%%%%%%%
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

    pipeline.fit(data_handler.df, y_surv)

    transformed_data = pipeline[:-1].transform(data_handler.df)
    print(transformed_data.shape)
    

