'''
tuned_rf(...) : figures out the best settings for random forests. Splits the training data
                from nested_cv intro three folds and uses them to evaluate settings. refits model with best metrics
nested_cv(...) :    divides data into k folds and uses trainingfold to figure settings
                    out, saves the different settings over the 5 fold and gives a more reliable metric score
tuned_gb(...) : similr to tuned_rf but for gradient boosting so different settings
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.inspection import permutation_importance
from linreg_func import metrics

def tuned_rf(n_trees=300, seed=12):
    param_grid = {"min_samples_leaf": [2, 4, 5],
                    "max_features":     [1, 2, 3]}
    
    forest = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    inner_folds = KFold(n_splits=3, shuffle=True, random_state=seed)

    return GridSearchCV(forest, param_grid, cv=inner_folds, scoring="r2")

def tuned_gb(max_trees=1000, seed=12):
    param_grid = {"learning_rate":    [0.05, 0.1],
                 "max_leaf_nodes":   [4, 7, 15],
                 "min_samples_leaf": [10, 20]}
 
    boosting = HistGradientBoostingRegressor(max_iter=max_trees, early_stopping=True,
                                             validation_fraction=0.1, n_iter_no_change=20,
                                             random_state=seed)
    inner_folds = KFold(n_splits=3, shuffle=True, random_state=seed)
 
    return GridSearchCV(boosting, param_grid, cv=inner_folds, scoring="r2")

def nested_cv(df, model, features, target="log_price", n_folds=5, seed=12):
    X = df[features]
    y = df[target]
 
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
 
    fold_scores = []
    fold_settings = []
    importance_rows = []

    for train_idx, test_idx in kf.split(X):
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
 
        model.fit(X_train, y_train)
 
        y_pred = model.predict(X_test)
        fold_scores.append(metrics(y_test, y_pred))
 
        # remeber settings chosen for this fold
        fold_settings.append(model.best_params_)

        # 5. Permutation importance on the test fold (shuffle each feature 5 times)
        perm = permutation_importance(model, X_test, y_test, scoring="r2",
                                      n_repeats=5, random_state=seed)
        for feature, value in zip(features, perm.importances_mean):
            importance_rows.append({"method": "permutation",
                                    "feature": feature,
                                    "importance": value})
 
        # 6. Built-in importance, if the model has it
        best_model = model.best_estimator_
        if hasattr(best_model, "feature_importances_"):
            for feature, value in zip(features, best_model.feature_importances_):
                importance_rows.append({"method": "builtin",
                                        "feature": feature,
                                        "importance": value})
 
    scores = pd.DataFrame(fold_scores)
    result = {
        "n":        len(df),
        "r2":       scores["r2"].mean(),   "r2_std":   scores["r2"].std(),
        "rmse":     scores["rmse"].mean(), "rmse_std": scores["rmse"].std(),
        "mae":      scores["mae"].mean(),  "mae_std":  scores["mae"].std(),
    }

    #mean and std for importance over each
    importance = (pd.DataFrame(importance_rows)
                    .groupby(["method", "feature"], sort=False)["importance"]
                    .agg(["mean", "std"])
                    .reset_index())
    
    return result, pd.DataFrame(fold_settings), importance
