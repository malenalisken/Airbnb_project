import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def adjusted_r2(r2, n, p):
    """Adjusted R²: penalises R² for the number of features p, given n observations."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def price_metrics(y_log_true, y_log_pred, n_features=1, adjusted=True):
    """Adjusted R² (or plain R² if adjusted=False) on log scale,
    RMSE and MAE in kroner."""
    price_true, price_pred = np.exp(y_log_true), np.exp(y_log_pred)

    r2 = r2_score(y_log_true, y_log_pred)
    r2_name = "r2"
    if adjusted:
        r2 = adjusted_r2(r2, len(y_log_true), n_features)
        r2_name = "r2_adj"

    return {
        r2_name: r2,
        "rmse":  np.sqrt(mean_squared_error(price_true, price_pred)),
        "mae":   mean_absolute_error(price_true, price_pred),
    }


def cross_validate(df, model, features, target="log_price", n_folds=5,
                   seed=42, adjusted=True):
    """K-fold CV of any model, on any listings, with any feature columns."""
    X, y = df[features], df[target]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_scores = []
    for train_idx, test_idx in kf.split(X):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[test_idx])
        fold_scores.append(price_metrics(y.iloc[test_idx], pred,
                                         len(features), adjusted))

    scores = pd.DataFrame(fold_scores)
    return {"n": len(df),
            **scores.mean().to_dict(),
            **scores.std().add_suffix("_std").to_dict()}


def backward_selection(df, model, candidates, target="log_price", n_folds=5):
    """Backward elimination: start with every candidate, then remove one at a
    time until r2 adjusted is no longer improved
 
    candidates: a list of columns, OR a dict {name: [columns]} so that several
                columns (e.g. all dummies) are removed or kept together.
    """
    if isinstance(candidates, dict):
        groups = dict(candidates)
    else:
        groups = {c: [c] for c in candidates}       # plain list: one column each
 
    def columns_of(names):
        return [col for name in names for col in groups[name]]
 
    # Step 0: the full model
    selected = list(groups)
    best_scores = cross_validate(df, model, columns_of(selected), target, n_folds)
    best_score = best_scores["r2_adj"]
    history = [{"step": 0, "removed": None, "n_remaining": len(selected),
                "score": best_score}]
 
    while len(selected) > 1:
        trials = {}
        for name in selected:
            remaining = [s for s in selected if s != name]
            trials[name] = cross_validate(df, model, columns_of(remaining),
                                          target, n_folds)
 
        drop_name = max(trials, key=lambda n: trials[n]["r2_adj"])
        new_score = trials[drop_name]["r2_adj"]
 
        if new_score < best_score:          # every removal hurts: stop
            break
 
        selected.remove(drop_name)
        best_score, best_scores = new_score, trials[drop_name]
        history.append({"step": len(history), "removed": drop_name,
                        "n_remaining": len(selected), "score": new_score})
 
    return selected, best_scores, pd.DataFrame(history)
 

def nested_backward_selection(df, model, candidates, target="log_price",
                              outer_folds=5, seed=42):
    """
    Split data using kfold keeping one fold for testing and the other folds for
    finding model using backwards selection. Model metrics is then calculated 
    by testing on outer fold.
 
    Returns
      folds     : one row per outer fold (what was selected, inner vs outer score)
    """
    if isinstance(candidates, dict):
        groups = dict(candidates)
    else:
        groups = {c: [c] for c in candidates}
 
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
 
    fold_rows, kept = [], []
    for i, (train_idx, test_idx) in enumerate(outer.split(df), start=1):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
 
        # Inner loop: choose variables using the training part only
        selected, inner_scores, _ = backward_selection(train, model, groups,
                                                       target)
        cols = [col for name in selected for col in groups[name]]
 
        # Refit on the full outer training part, score on the outer test fold
        fitted = clone(model).fit(train[cols], train[target])
        pred = fitted.predict(test[cols])
        outer_scores = price_metrics(test[target], pred, len(cols))
 
        kept.append(selected)
        fold_rows.append({"fold": i,
                          "n_selected": len(selected),
                          "selected_features": ", ".join(selected),
                          "inner_r2_adj": inner_scores["r2_adj"],
                          **outer_scores})
 
    folds = pd.DataFrame(fold_rows)
    return folds



RF_PARAM_GRID = {"min_samples_leaf": [10, 20],
                 "max_features":     [2,4]}

def tuned_rf(param_grid=RF_PARAM_GRID, n_trees=300, inner_folds=3, seed=42):
    """Random forest that tunes its settings with inner CV each time it is fitted."""
    return GridSearchCV(
        RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1),
        param_grid,
        cv=KFold(n_splits=inner_folds, shuffle=True, random_state=seed),
        scoring="r2")

def nested_cv_tuned(df, search, features, target="log_price", outer_folds=5, seed=42):
    """Nested CV for a self-tuning model (e.g. tuned_rf()): tuning happens inside
    each outer training part, scoring on the outer test fold.

    Returns one row per outer fold with the chosen settings and its scores."""
    X, y = df[features], df[target]
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    fold_rows = []
    for i, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
        fitted = clone(search).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[test_idx])
        fold_rows.append({"fold": i,
                          **fitted.best_params_,
                          "inner_r2": fitted.best_score_,
                          **price_metrics(y.iloc[test_idx], pred, adjusted=False)})

    return pd.DataFrame(fold_rows)

GB_PARAM_GRID = {"learning_rate":    [0.5, 0.1],
                 "max_leaf_nodes":   [10, 25],
                 "min_samples_leaf": [20, 50]}

def tuned_gb(param_grid=GB_PARAM_GRID, max_trees=500, inner_folds=3, seed=42):
    """Gradient boosting that tunes its settings with inner CV each time it is fitted.
    Early stopping decides the number of trees automatically."""
    return GridSearchCV(
        HistGradientBoostingRegressor(max_iter=max_trees, early_stopping=True,
                                      validation_fraction=0.1, n_iter_no_change=20,
                                      random_state=seed),
        param_grid,
        cv=KFold(n_splits=inner_folds, shuffle=True, random_state=seed),
        scoring="r2")