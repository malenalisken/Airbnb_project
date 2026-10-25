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

#start by evaluating each variable aganst price_log
df['neighbourhood_grouped'] = df['neighbourhood_cleansed'].replace({
    "Stovner": "Stovner/Grorud/Alna",
    "Grorud": "Stovner/Grorud/Alna",
    "Alna": "Stovner/Grorud/Alna",
})
df_area = df[df['neighbourhood_cleansed'] != 'Marka'].copy()

#-------------------------------------------------------------------------------
# linreg model per listing variable with target log_price
#-------------------------------------------------------------------------------
metrics = []
for area, obs in df_area.groupby('neighbourhood_grouped'):
    for var in listing_vars:
        scores = cross_validate(obs, LinearRegression(), [var], adjusted = False)
        metrics.append({"area": area, "variable": var, **scores})

single_var_linreg_df = (pd.DataFrame(metrics)
                .sort_values(["area", "r2"], ascending=[True, False])
                .reset_index(drop=True))
print(single_var_linreg_df.round(3))
single_var_linreg_df.to_csv(RESULTS / "area_single_variable.csv", index=False)


#-------------------------------------------------------------------------------
# linreg model with all listing vars with target log_price
#-------------------------------------------------------------------------------
metrics = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    scores = cross_validate(group, LinearRegression(), listing_vars)
    metrics.append({"area": area, **scores})
    
all_var_linreg_df = (pd.DataFrame(metrics)
                .sort_values("r2_adj", ascending=False)
                .reset_index(drop=True))
print(f'\n{all_var_linreg_df.round(3)}')
all_var_linreg_df.to_csv(RESULTS / "area_all_variable.csv", index=False)


#-------------------------------------------------------------------------------
# Backward linear regression, finding best model
#-------------------------------------------------------------------------------
 
metrics = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    model_vars, scores, history = backward_selection(group, LinearRegression(), listing_vars)
    removed = [f for f in listing_vars if f not in model_vars]
    metrics.append({"area": area,
                    "n_selected": len(model_vars),
                    "selected_features": ", ".join(model_vars),
                    "removed_features": ", ".join(removed),
                    **scores})
 
backward_linreg_df = (pd.DataFrame(metrics)
                .sort_values("r2_adj", ascending=False)
                .reset_index(drop=True))
print(backward_linreg_df.round(3))
backward_linreg_df.to_csv(RESULTS / "area_backward_selection.csv", index=False)

#-------------------------------------------------------------------------------
# Nested CV backwards regression: estimate of how well backward selection performs
#-------------------------------------------------------------------------------
 
all_folds = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    folds = nested_backward_selection(group, LinearRegression(), listing_vars)
    all_folds.append(folds.assign(area=area, n=len(group)))
 
folds_df = pd.concat(all_folds).reset_index(drop=True)
print(f'\nNested CV per area and outer fold:\n{folds_df.round(3)}')
folds_df.to_csv(RESULTS / "area_nested_folds.csv", index=False)

#-------------------------------------------------------------------------------
# Random forest with all listing vars, nested CV (tuning inside each outer fold)
#-------------------------------------------------------------------------------
all_folds = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    folds = nested_cv_tuned(group, tuned_rf(), listing_vars)
    all_folds.append(folds.assign(area=area, n=len(group)))

rf_folds_df = pd.concat(all_folds).reset_index(drop=True)
print(f'\nRandom forest nested CV per area and outer fold:\n{rf_folds_df.round(3)}')
rf_folds_df.to_csv(RESULTS / "area_rf_nested_folds.csv", index=False)

# One row per area: mean and std of the outer-fold scores
rf_summary_df = (rf_folds_df.groupby("area")
                            .agg(n=("n", "first"),
                                 r2=("r2", "mean"),     r2_std=("r2", "std"),
                                 rmse=("rmse", "mean"), rmse_std=("rmse", "std"),
                                 mae=("mae", "mean"),   mae_std=("mae", "std"))
                            .sort_values("r2", ascending=False)
                            .reset_index())
print(f'\nRandom forest per area (nested CV):\n{rf_summary_df.round(3)}')
rf_summary_df.to_csv(RESULTS / "area_rf_summary.csv", index=False)

#-------------------------------------------------------------------------------
# Gradient boosting with all listing vars, nested CV per area
#-------------------------------------------------------------------------------
all_folds = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    folds = nested_cv_tuned(group, tuned_gb(), listing_vars)
    all_folds.append(folds.assign(area=area, n=len(group)))

gb_folds_df = pd.concat(all_folds).reset_index(drop=True)
print(f'\nGradient boosting nested CV per area and outer fold:\n{gb_folds_df.round(3)}')
gb_folds_df.to_csv(RESULTS / "area_gb_nested_folds.csv", index=False)

gb_summary_df = (gb_folds_df.groupby("area")
                            .agg(n=("n", "first"),
                                 r2=("r2", "mean"),     r2_std=("r2", "std"),
                                 rmse=("rmse", "mean"), rmse_std=("rmse", "std"),
                                 mae=("mae", "mean"),   mae_std=("mae", "std"))
                            .sort_values("r2", ascending=False)
                            .reset_index())
print(f'\nGradient boosting per area (nested CV):\n{gb_summary_df.round(3)}')
gb_summary_df.to_csv(RESULTS / "area_gb_summary.csv", index=False)