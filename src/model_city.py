import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree

from evaluate import cross_validate, backward_selection, nested_backward_selection, tuned_rf, nested_cv_tuned, tuned_gb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
RESULTS = PROJECT_ROOT / "results"
df = pd.read_csv(PROCESSED_DATA)

features = ['host_is_superhost', 'is_house', 'accommodates', 'bedrooms',
            'bathrooms', 'minimum_nights_log', 'has_rating', 'weighted_rating']

# Same grouping as the area models, and Marka dropped,
# so city and area models are evaluated on the same listings
df['neighbourhood_grouped'] = df['neighbourhood_cleansed'].replace({
    "Stovner": "Stovner/Grorud/Alna",
    "Grorud": "Stovner/Grorud/Alna",
    "Alna": "Stovner/Grorud/Alna",
})
df_city = df[df['neighbourhood_grouped'] != 'Marka'].copy()

# Neighbourhood dummies, Sentrum = baseline (no column).
# Any dummy_ columns from cleaning are removed first so they aren't duplicated.
df_city = df_city.drop(columns=df_city.filter(like="dummy_").columns)
dummies = pd.get_dummies(df_city['neighbourhood_grouped'], prefix="dummy", dtype=int)
dummies = dummies.drop(columns="dummy_Sentrum")
df_city = pd.concat([df_city, dummies], axis=1)

# Every candidate variable, with the columns that belong to it.
# The dummies together are ONE variable: neighbourhood.
candidates = {feature: [feature] for feature in features}
candidates["neighbourhood"] = dummies.columns.tolist()

#-------------------------------------------------------------------------------
# Linear regression per variable against log_price
#-------------------------------------------------------------------------------
results = []
for name, cols in candidates.items():
    # plain R² for one column, adjusted R² for neighbourhood (many columns)
    scores = cross_validate(df_city, LinearRegression(), cols,
                            adjusted=len(cols) > 1)
    results.append({"variable": name, "n_columns": len(cols), **scores})

results_df = pd.DataFrame(results)
results_df["sort_key"] = results_df["r2"].fillna(results_df["r2_adj"])
results_df = (results_df.sort_values("sort_key", ascending=False)
                        .drop(columns="sort_key")
                        .reset_index(drop=True))
print(results_df.round(3))
results_df.to_csv(RESULTS / "city_single_variable.csv", index=False)

#-------------------------------------------------------------------------------
# Linear regression all variables (features + neighbourhood) against log_price
#-------------------------------------------------------------------------------
all_columns = [col for cols in candidates.values() for col in cols]
scores = cross_validate(df_city, LinearRegression(), all_columns)

results_df = pd.DataFrame([{"n_columns": len(all_columns), **scores}])
print(f'\n{results_df.round(3)}')
results_df.to_csv(RESULTS / "city_all_variable.csv", index=False)

#-------------------------------------------------------------------------------
# Backward linear regression, finding best model
#-------------------------------------------------------------------------------
selected, scores, history = backward_selection(df_city, LinearRegression(), candidates)
removed = [name for name in candidates if name not in selected]
 
print("\nBackward elimination steps:")
print(history.round(3))
 
results_df = pd.DataFrame([{"n_selected": len(selected),
                            "selected_features": ", ".join(selected),
                            "removed_features": ", ".join(removed),
                            **scores}])
print(f'\n{results_df.round(3)}')
results_df.to_csv(RESULTS / "city_backward_selection.csv", index=False)

#-------------------------------------------------------------------------------
# Nested CV backwards regression: estimate of how well backward selection performs
#-------------------------------------------------------------------------------
folds = nested_backward_selection(df_city, LinearRegression(), candidates)
folds["n"] = len(df_city)

print(f'\nNested CV per outer fold:\n{folds.round(3)}')
folds.to_csv(RESULTS / "city_nested_folds.csv", index=False)

#-------------------------------------------------------------------------------
# Random forest with all variables (features + neighbourhood), nested CV
#-------------------------------------------------------------------------------
rf_folds = nested_cv_tuned(df_city, tuned_rf(), all_columns)
rf_folds["n"] = len(df_city)

print(f'\nRandom forest nested CV per outer fold:\n{rf_folds.round(3)}')
rf_folds.to_csv(RESULTS / "city_rf_nested_folds.csv", index=False)

# One summary row: mean and std of the outer-fold scores
rf_summary = pd.DataFrame([{
    "n":        len(df_city),
    "r2":       rf_folds["r2"].mean(),   "r2_std":   rf_folds["r2"].std(),
    "rmse":     rf_folds["rmse"].mean(), "rmse_std": rf_folds["rmse"].std(),
    "mae":      rf_folds["mae"].mean(),  "mae_std":  rf_folds["mae"].std(),
}])
print(f'\nRandom forest city (nested CV):\n{rf_summary.round(3)}')
rf_summary.to_csv(RESULTS / "city_rf_summary.csv", index=False)


#-------------------------------------------------------------------------------
# Gradient boosting with all variables (features + neighbourhood), nested CV
#-------------------------------------------------------------------------------
gb_folds = nested_cv_tuned(df_city, tuned_gb(), all_columns)
gb_folds["n"] = len(df_city)

print(f'\nGradient boosting nested CV per outer fold:\n{gb_folds.round(3)}')
gb_folds.to_csv(RESULTS / "city_gb_nested_folds.csv", index=False)

gb_summary = pd.DataFrame([{
    "n":    len(df_city),
    "r2":   gb_folds["r2"].mean(),   "r2_std":   gb_folds["r2"].std(),
    "rmse": gb_folds["rmse"].mean(), "rmse_std": gb_folds["rmse"].std(),
    "mae":  gb_folds["mae"].mean(),  "mae_std":  gb_folds["mae"].std(),
}])
print(f'\nGradient boosting city (nested CV):\n{gb_summary.round(3)}')
gb_summary.to_csv(RESULTS / "city_gb_summary.csv", index=False)