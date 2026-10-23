import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score




project_root = Path(__file__).resolve().parents[1]
data_path = project_root / 'data' / 'processed' / 'listings_model.csv'
df_model = pd.read_csv(data_path).copy()

#group together stovner grorud and alna together to get more datapoints
print(df_model.columns.tolist())
df_model["neighbourhood_grouped"] = df_model["neighbourhood_cleansed"].replace({
    "Stovner": "Stovner/Grorud/Alna",
    "Grorud": "Stovner/Grorud/Alna",
    "Alna": "Stovner/Grorud/Alna",
})
# Dropping Marka: too few observations, and its listings likely differ from the rest of Oslo
df_model = df_model[df_model["neighbourhood_grouped"] != "Marka"].copy()

print(df_model["neighbourhood_grouped"].value_counts())



#-----------------------------------------------------------------------------------------------------------------------
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Finding linear regression models per independent variable for each area
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#-----------------------------------------------------------------------------------------------------------------------



independent_var = ['host_is_superhost', 'is_house', 'accommodates', 'bedrooms','bathrooms', 
                   'minimum_nights', 'has_rating', 'weighted_rating', 'accommodates_log', 'minimum_nights_log']

target = "log_price"
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def kfold_simple_regression(data, x_var, y_var, kf):
    """Simple linear regression y ~ x with k-fold CV.
    Returns test R² and coefficients averaged over the folds."""
    X = data[[x_var]].values
    y = data[y_var].values

    fold_r2, fold_slope, fold_intercept = [], [], []
    for train_idx, test_idx in kf.split(X):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        fold_r2.append(r2_score(y[test_idx], y_pred))
        fold_slope.append(model.coef_[0])
        fold_intercept.append(model.intercept_)

    return {
        "n": len(data),
        "Slope": np.mean(fold_slope),
        "Slope std": np.std(fold_slope),          # how much the slope varies between folds
        "CV R2": np.mean(fold_r2),
        "CV R2 std": np.std(fold_r2),
    }


# loop over each area
results = []
for area, g in df_model.groupby("neighbourhood_grouped"):
    for var in independent_var:
        # skip if the variable has the same value for every listing in this area
        if g[var].nunique() < 2:
            continue
        res = kfold_simple_regression(g, var, target, kf)
        results.append({"Area": area, "Variable": var, **res})

results = pd.DataFrame(results)

#print as table in terminal so it is easier to look at
for area, table in results.groupby("Area"):
    table = table.drop(columns="Area").sort_values("CV R2", ascending=False)

    print(f"\n{'=' * 80}")
    print(f"  {area}  (n = {table['n'].iloc[0]})")
    print(f"{'=' * 80}")
    print(table.drop(columns="n").to_string(
        index=False,
        formatters={
            "Intercept": "{:.3f}".format,
            "Slope": "{:.4f}".format,
            "Slope std": "{:.4f}".format,
            "Effect %": "{:+.1f}%".format,
            "CV R2": "{:.3f}".format,
            "CV R2 std": "{:.3f}".format,
        },
    ))

#-----------------------------------------------------------------------------------------------------------------------
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Finding multiple linear regression with all chosen variables for each area
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#-----------------------------------------------------------------------------------------------------------------------


chosen_var = ['host_is_superhost', 'is_house', 'accommodates_log', 'bedrooms',
              'bathrooms', 'minimum_nights_log', 'has_rating', 'weighted_rating']


def kfold_multi_regression(data, x_vars, y_var, kf):
    """Multiple linear regression y ~ x1 + x2 + ... with k-fold CV.
    Returns adjusted test R² averaged over the folds."""
    X = data[x_vars].values
    y = data[y_var].values
    p = X.shape[1]                                # number of independent variables

    fold_r2_adj = []
    for train_idx, test_idx in kf.split(X):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        r2 = r2_score(y[test_idx], y_pred)
        n_test = len(test_idx)
        fold_r2_adj.append(1 - (1 - r2) * (n_test - 1) / (n_test - p - 1))

    return {
        "n": len(data),
        "CV R2 adj": np.mean(fold_r2_adj),
        "CV R2 adj std": np.std(fold_r2_adj),    # how much R2 adj varies between folds
    }


# loop over each area
multi_results = []
for area, g in df_model.groupby("neighbourhood_grouped"):
    res = kfold_multi_regression(g, chosen_var, target, kf)
    multi_results.append({"Area": area, **res})

multi_results = pd.DataFrame(multi_results).sort_values("CV R2 adj", ascending=False)

#print as table in terminal so it is easier to look at
print(f"\n{'=' * 80}")
print("  Multiple linear regression with all chosen variables")
print(f"{'=' * 80}")
print(multi_results.to_string(
    index=False,
    formatters={
        "CV R2 adj": "{:.3f}".format,
        "CV R2 adj std": "{:.3f}".format,
    },
))


#-----------------------------------------------------------------------------------------------------------------------
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Finding multiple linear regression with forward regression analisys
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Finding the best combination of variables for each area with forward selection
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#-----------------------------------------------------------------------------------------------------------------------



def forward_selection(data, candidates, y_var, kf):
    """Add one variable at a time, choosing the one that increases CV R2 adj the most.
    Stops when no remaining variable improves CV R2 adj."""
    selected = []
    remaining = [v for v in candidates if data[v].nunique() > 1]
    best = {"CV R2 adj": -np.inf, "CV R2 adj std": np.nan}

    while remaining:
        # test adding each remaining variable to the current model
        scores = {var: kfold_multi_regression(data, selected + [var], y_var, kf) for var in remaining}
        best_var = max(scores, key=lambda v: scores[v]["CV R2 adj"])

        # stop if the best new variable does not improve the model
        if scores[best_var]["CV R2 adj"] <= best["CV R2 adj"]:
            break

        selected.append(best_var)
        remaining.remove(best_var)
        best = scores[best_var]

    return {
        "n": len(data),
        "Number of vars": len(selected),
        "CV R2 adj": best["CV R2 adj"],
        "CV R2 adj std": best["CV R2 adj std"],
        "Variables": ", ".join(selected),        # in the order they were added
    }


# loop over each area
forward_results = []
for area, g in df_model.groupby("neighbourhood_grouped"):
    res = forward_selection(g, chosen_var, target, kf)
    forward_results.append({"Area": area, **res})

forward_results = pd.DataFrame(forward_results).sort_values("CV R2 adj", ascending=False)

#print as table in terminal so it is easier to look at
print(f"\n{'=' * 80}")
print("  Forward selection: best combination of variables per area")
print(f"{'=' * 80}")
print(forward_results.to_string(
    index=False,
    formatters={
        "CV R2 adj": "{:.3f}".format,
        "CV R2 adj std": "{:.3f}".format,
    }
))