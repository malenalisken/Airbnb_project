'''
metrics(...) :  computes the scores fir evaluation of 
                model performance (r^2, RMSE and MAE)
cross_validate(...) :   calculated average metrics of model over k folds and 
                        returns scores and standard deviations
coefficients(...) :     coefficients for lin reg model on entire dataset
backwards_selection(...) :  starts with a model that has all variables, removed one 
                            variable at a time until the r2 score is no longer imporved
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def metrics(y_log_true, y_log_pred):
    price_true = np.exp(y_log_true)
    price_pred = np.exp(y_log_pred)

    return{
        "r2":   r2_score(y_log_true, y_log_pred),
        "rmse": np.sqrt(mean_squared_error(price_true, price_pred)),
        "mae":  mean_absolute_error(price_true, price_pred)
    }

def cross_validate(df, features, target = 'log_price', n_folds = 5, seed = 12):
    X = df[features]
    Y = df[target]

    kf = KFold(n_folds, shuffle = True, random_state = seed)
    fold_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)
        fold_scores.append(metrics(y_test, y_pred))

        scores = pd.DataFrame(fold_scores)

    return{
        "n":        len(df),
        "r2":       scores["r2"].mean(),   "r2_std":   scores["r2"].std(),
        "rmse":     scores["rmse"].mean(), "rmse_std": scores["rmse"].std(),
        "mae":      scores["mae"].mean(),  "mae_std":  scores["mae"].std(),
    
    }

def coefficients(df, features, target = 'log_price'):
    #features must be passed a list even if only one var
    model = LinearRegression().fit(df[features], df[target])

    coefs = {}
    for name, value in zip(features, model.coef_):
        coefs[name] = value

    return coefs

def backwards_selection(df, variables, target = 'log_price'):
    '''
    Returns selected vars, scores of final model and a history of which vars were removed
    '''
    selected = list(variables) 

    cols = []
    for var in selected:
        cols += variables[var]

    best_scores = cross_validate(df, cols, target)
 
    history = [{"step": 0, "removed": None,
                "n_variables": len(selected), "r2": best_scores["r2"]}]
    
    while len(selected) > 1:
    
        remove_var = None
        best_trial = None

        for var in selected:
            cols = []
            for other in selected:
                if other != var:
                    cols += variables[other]
            scores = cross_validate(df, cols, target)
 
            if best_trial is None or scores["r2"] > best_trial["r2"]:
                remove_var = var
                best_trial = scores
 
        # stop if removal makes model worse
        if best_trial["r2"] < best_scores["r2"]:
            break
 
        # Otherwise remove that variable and continue
        selected.remove(remove_var)
        best_scores = best_trial
        history.append({"step": len(history), "removed": remove_var,
                        "n_variables": len(selected), "r2": best_scores["r2"]})
 
    return selected, best_scores, pd.DataFrame(history)