import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LinearRegression

from evaluate import cross_validate, backward_selection, nested_backward_selection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
RESULTS = PROJECT_ROOT / "results"
df = pd.read_csv(PROCESSED_DATA)

features = ['host_is_superhost', 'is_house', 'accommodates', 'bedrooms',
            'bathrooms', 'minimum_nights_log', 'has_rating', 'weighted_rating']

#start by evaluating each variable aganst price_log
df['neighbourhood_grouped'] = df['neighbourhood_cleansed'].replace({
    "Stovner": "Stovner/Grorud/Alna",
    "Grorud": "Stovner/Grorud/Alna",
    "Alna": "Stovner/Grorud/Alna",
})
df_area = df[df['neighbourhood_cleansed'] != 'Marka'].copy()

#-------------------------------------------------------------------------------
# Linear regression per variable against log_price
#-------------------------------------------------------------------------------
results = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    for feature in features:
        scores = cross_validate(group, LinearRegression(), [feature], adjusted = False)
        results.append({"area": area, "variable": feature, **scores})

results_df = (pd.DataFrame(results)
                .sort_values(["area", "r2"], ascending=[True, False])
                .reset_index(drop=True))
print(results_df.round(3))
results_df.to_csv(RESULTS / "area_single_variable.csv", index=False)

#-------------------------------------------------------------------------------
# Linear regression all variables against log_price
#-------------------------------------------------------------------------------
results = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    scores = cross_validate(group, LinearRegression(), features)
    results.append({"area": area, **scores})
    
results_df = (pd.DataFrame(results)
                .sort_values("r2_adj", ascending=False)
                .reset_index(drop=True))
print(f'\n{results_df.round(3)}')
results_df.to_csv(RESULTS / "area_all_variable.csv", index=False)


#-------------------------------------------------------------------------------
# Backward linear regression, finding best model
#-------------------------------------------------------------------------------
 
results = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    selected, scores, history = backward_selection(group, LinearRegression(), features)
    removed = [f for f in features if f not in selected]
    results.append({"area": area,
                    "n_selected": len(selected),
                    "selected_features": ", ".join(selected),
                    "removed_features": ", ".join(removed),
                    **scores})
 
results_df = (pd.DataFrame(results)
                .sort_values("r2_adj", ascending=False)
                .reset_index(drop=True))
print(results_df.round(3))
results_df.to_csv(RESULTS / "area_backward_selection.csv", index=False)

#-------------------------------------------------------------------------------
# Nested CV backwards regression: estimate of how well backward selection performs
#-------------------------------------------------------------------------------
 
all_folds = []
for area, group in df_area.groupby('neighbourhood_grouped'):
    folds = nested_backward_selection(group, LinearRegression(), features)
    all_folds.append(folds.assign(area=area, n=len(group)))
 
folds_df = pd.concat(all_folds).reset_index(drop=True)
print(f'\nNested CV per area and outer fold:\n{folds_df.round(3)}')
folds_df.to_csv(RESULTS / "area_nested_folds.csv", index=False)