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
from DataManagement import DataHandler


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

        self.data_handler.aggregate()
        self.data_handler.categorize()

        X = self.data_handler.df
        y = self.data_handler.y

        # Float preprocessing
        float_pipe = Pipeline(steps=[
            ("impute_num", SimpleImputer(strategy="median"))
        ])

        # Categorial Preprocessing
        cat_pipe = Pipeline(steps=[
            ("impute_cat", KNNImputer(
                n_neighbors=5,
                weights="distance"
            )),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ))
        ])

        # columns transformer
        col_trans = ColumnTransformer(
            transformers=[
                ("num", float_pipe, self.data_handler.float_cols),
                ("cat", cat_pipe, self.data_handler.categorical_cols)

            ],
            remainders='passthrough'
        )

        # return full Pipeline
        return Pipeline([
            ("drop_missing", DropMissingTransformer(threshold=0.2)),
            ('column_transformer', col_trans),
        ])
    



# %%%%%%%%%%%%% === MAIN TEST === %%%%%%%%%%%%%
if __name__ == "__main__":
    pass