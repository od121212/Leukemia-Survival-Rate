from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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



# ========= Abstract Base Class for Model Pipelines =========
class ModelPipeline(ABC):

    @abstractmethod
    def build_pipeline(self, drop_missing_threshold=0.2) -> Pipeline:
        pass


class PipelineFactory:

    @staticmethod
    def get_pipeline(model_type: str) -> ModelPipeline:
        if model_type == 'neural_network':
            pass
        elif model_type == 'linear_regression':
            pass
        else:
            return DefaultPipeLine()  # Return a default pipeline for unknown types
        


class DefaultPipeLine(ModelPipeline, DataHandler):

    def __init__(self):
        super().__init__()

    def build_pipeline(self, data_handler: DataHandler, drop_missing_threshold=0.2) -> Pipeline:

        # Passer DataHandler en héritage pour faciliter

        # encoder = OneHotEncoder(
        #     data_handler.categorical_cols,

        # )


# Example
        # return Pipeline([
        #     ("drop_missing", DropMissingTransformer(threshold=0.2)),
        # ])
    
        pass