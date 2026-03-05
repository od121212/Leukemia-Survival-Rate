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

class FeatureRatioCreator(BaseEstimator, TransformerMixin):
    """
    Create derived ratio features after imputation and preprocessing.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # no fitting required

    def transform(self, X, y=None):
        X = X.copy()

        # --- Ratios ---
        if {"ANC", "WBC"}.issubset(X.columns):
            X["anc_ratio"] = X["ANC"] / X["WBC"]

        if {"MONOCYTES", "WBC"}.issubset(X.columns):
            X["mono_ratio"] = X["MONOCYTES"] / X["WBC"]

        if {"ANC", "MONOCYTES"}.issubset(X.columns):
            X["anc_mono_ratio"] = X["ANC"] / X["MONOCYTES"]

        if {"PLT", "WBC"}.issubset(X.columns):
            X["plt_wbc_ratio"] = X["PLT"] / X["WBC"]

        # --- Log transforms ---
        if "WBC" in X.columns:
            X["log_WBC"] = np.log1p(X["WBC"])

        if "BM_BLAST" in X.columns:
            X["log_BM_BLAST"] = np.log1p(X["BM_BLAST"])

        # --- Interactions ---
        if {"log_WBC", "BM_BLAST"}.issubset(X.columns):
            X["blast_burden"] = X["log_WBC"] * (X["BM_BLAST"] / 100)

        if {"BM_BLAST", "max_vaf"}.issubset(X.columns):
            X["blast_clone_interaction"] = X["BM_BLAST"] * X["max_vaf"]

        if {"cyto_complex", "nb_mutations"}.issubset(X.columns):
            X["complex_mutation"] = (X["cyto_complex"].fillna(0) > 0).astype(int) * X["nb_mutations"]

        return X

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

    def __init__(self, *args):
        pass

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

    def __init__(
            self, 
            prep_:tuple[pd.DataFrame, pd.DataFrame, list, list, list], 
            *args
            ):
        super().__init__(*args)
        self.prepared_data = prep_

    def build_pipeline(self) -> Pipeline:

        # return prepared data set copies (non-mutating)
        X, y, float_cols, categorical_cols, binary_cols = self.prepared_data

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
                ("num", float_pipe, float_cols),
                ("cat", multi_cat_pipe, categorical_cols),
                ("binary_cat", binary_pipe, binary_cols)
            ],
            remainder='passthrough'
        )
        col_trans.set_output(transform="pandas")

        # === MODEL ===

        model = RandomSurvivalForest(
            n_jobs=-1,
            random_state=42,
            bootstrap=True,
            oob_score=False
        )


        # return full Pipeline
        return Pipeline([
            ("drop_missing", DropMissingTransformer(threshold=0.2)),
            ('column_transformer', col_trans),
            ('feature_ratios', FeatureRatioCreator()),
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
    prepared_data = data_handler.prepare()
    pipeline_builder = DefaultPipeline(prepared_data)
    pipeline = pipeline_builder.build_pipeline()


    y_surv = Surv.from_dataframe(
        event='OS_STATUS',   # 1 = event, 0 = censored
        time='OS_YEARS',
        data=prepared_data[1]
    )

    pipeline.fit(prepared_data[0], y_surv)

    transformed_data = pipeline[:-1].transform(prepared_data[0])
    print(transformed_data.shape)
    

