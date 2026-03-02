import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

def cindex_score(y_true, risk_scores):
    return concordance_index_censored(
        y_true["OS_STATUS"],
        y_true["OS_YEARS"],
        risk_scores
    )[0]

def learning_curve_analysis(pipeline,X,y):

    cindex_scorer = make_scorer(cindex_score, greater_is_better=True)

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=cindex_scorer,
        cv=5,
        n_jobs=-1,
        verbose=3
    )

    train_sizes = np.linspace(0.1, 0.9, 6)
    train_scores = []
    val_scores = []

    for frac in train_sizes:

        X_sub, _, y_sub, _ = train_test_split(
            X,
            y,
            train_size=float(frac),
            random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_sub,
            y_sub,
            test_size=0.3,
            random_state=42
        )

        # refit grid search
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # risk predictions
        train_risk = best_model.predict(X_train)
        val_risk = best_model.predict(X_val)

        # C-index train
        train_cindex = concordance_index_censored(
            y_train["OS_STATUS"],
            y_train["OS_YEARS"],
            train_risk
        )[0]

        # C-index validation
        val_cindex = concordance_index_censored(
            y_val["OS_STATUS"],
            y_val["OS_YEARS"],
            val_risk
        )[0]

        train_scores.append(train_cindex)
        val_scores.append(val_cindex)

    plt.figure(figsize=(8, 5))

    plt.plot(train_sizes, train_scores, marker="o", label="Train C-index")
    plt.plot(train_sizes, val_scores, marker="o", label="Validation C-index")

    plt.xlabel("Training set fraction")
    plt.ylabel("C-index")
    plt.title("Learning Curve — Overfitting Diagnosis")

    plt.legend()
    plt.grid()
    plt.show()