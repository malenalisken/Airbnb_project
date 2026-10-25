import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
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
 