import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from evaluate import cross_validate, backward_selection, nested_backward_selection, tuned_rf, nested_cv_tuned, tuned_gb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
RESULTS = PROJECT_ROOT / "results"
df = pd.read_csv(PROCESSED_DATA)

listing_vars = ['host_is_superhost', 'is_house', 'accommodates', 'bedrooms',
            'bathrooms', 'minimum_nights_log', 'has_rating', 'weighted_rating']

# Grouping and Marka removal are done in cleaning.py
df_district = df.copy()

#-------------------------------------------------------------------------------
# linreg model per listing variable with target log_price
#-------------------------------------------------------------------------------
metrics = []
for district, obs in df_district.groupby('neighbourhood_grouped'):
    for var in listing_vars:
        scores = cross_validate(obs, LinearRegression(), [var], adjusted = False)
        metrics.append({"district": district, "variable": var, **scores})

single_var_linreg_df = (pd.DataFrame(metrics)
                .sort_values(["district", "r2"], ascending=[True, False])
                .reset_index(drop=True))
print(single_var_linreg_df.round(3))
single_var_linreg_df.to_csv(RESULTS / "district_single_variable.csv", index=False)


#-------------------------------------------------------------------------------
# linreg model with all listing vars with target log_price
#-------------------------------------------------------------------------------
metrics = []
for district, group in df_district.groupby('neighbourhood_grouped'):
    scores = cross_validate(group, LinearRegression(), listing_vars)
    metrics.append({"district": district, **scores})
    
all_var_linreg_df = (pd.DataFrame(metrics)
                .sort_values("r2_adj", ascending=False)
                .reset_index(drop=True))
print(f'\n{all_var_linreg_df.round(3)}')
all_var_linreg_df.to_csv(RESULTS / "district_all_variable.csv", index=False)


#-------------------------------------------------------------------------------
# Backward linear regression, finding best model
#-------------------------------------------------------------------------------
 
metrics = []
for district, group in df_district.groupby('neighbourhood_grouped'):
    model_vars, scores, history = backward_selection(group, LinearRegression(), listing_vars)
    removed = [f for f in listing_vars if f not in model_vars]
    metrics.append({"district": district,
                    "n_selected": len(model_vars),
                    "selected_features": ", ".join(model_vars),
                    "removed_features": ", ".join(removed),
                    **scores})
 
backward_linreg_df = (pd.DataFrame(metrics)
                .sort_values("r2_adj", ascending=False)
                .reset_index(drop=True))
print(backward_linreg_df.round(3))
backward_linreg_df.to_csv(RESULTS / "district_backward_selection.csv", index=False)

#-------------------------------------------------------------------------------
# Nested CV backwards regression: estimate of how well backward selection performs
#-------------------------------------------------------------------------------
 
all_folds = []
for district, group in df_district.groupby('neighbourhood_grouped'):
    folds = nested_backward_selection(group, LinearRegression(), listing_vars)
    all_folds.append(folds.assign(district=district, n=len(group)))
 
folds_df = pd.concat(all_folds).reset_index(drop=True)
print(f'\nNested CV per district and outer fold:\n{folds_df.round(3)}')
folds_df.to_csv(RESULTS / "district_nested_folds.csv", index=False)

#-------------------------------------------------------------------------------
# Random forest with all listing vars, nested CV (tuning inside each outer fold)
#-------------------------------------------------------------------------------
all_folds = []
for district, group in df_district.groupby('neighbourhood_grouped'):
    folds = nested_cv_tuned(group, tuned_rf(), listing_vars)
    all_folds.append(folds.assign(district=district, n=len(group)))

rf_folds_df = pd.concat(all_folds).reset_index(drop=True)
print(f'\nRandom forest nested CV per district and outer fold:\n{rf_folds_df.round(3)}')
rf_folds_df.to_csv(RESULTS / "district_rf_nested_folds.csv", index=False)

# One row per district: mean and std of the outer-fold scores
rf_summary_df = (rf_folds_df.groupby("district")
                            .agg(n=("n", "first"),
                                 r2=("r2", "mean"),     r2_std=("r2", "std"),
                                 rmse=("rmse", "mean"), rmse_std=("rmse", "std"),
                                 mae=("mae", "mean"),   mae_std=("mae", "std"))
                            .sort_values("r2", ascending=False)
                            .reset_index())
print(f'\nRandom forest per district (nested CV):\n{rf_summary_df.round(3)}')
rf_summary_df.to_csv(RESULTS / "district_rf_summary.csv", index=False)

#-------------------------------------------------------------------------------
# Gradient boosting with all listing vars, nested CV per district
#-------------------------------------------------------------------------------
all_folds = []
for district, group in df_district.groupby('neighbourhood_grouped'):
    folds = nested_cv_tuned(group, tuned_gb(), listing_vars)
    all_folds.append(folds.assign(district=district, n=len(group)))

gb_folds_df = pd.concat(all_folds).reset_index(drop=True)
print(f'\nGradient boosting nested CV per district and outer fold:\n{gb_folds_df.round(3)}')
gb_folds_df.to_csv(RESULTS / "district_gb_nested_folds.csv", index=False)

gb_summary_df = (gb_folds_df.groupby("district")
                            .agg(n=("n", "first"),
                                 r2=("r2", "mean"),     r2_std=("r2", "std"),
                                 rmse=("rmse", "mean"), rmse_std=("rmse", "std"),
                                 mae=("mae", "mean"),   mae_std=("mae", "std"))
                            .sort_values("r2", ascending=False)
                            .reset_index())
print(f'\nGradient boosting per district (nested CV):\n{gb_summary_df.round(3)}')
gb_summary_df.to_csv(RESULTS / "district_gb_summary.csv", index=False)