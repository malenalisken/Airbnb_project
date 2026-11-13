
import pandas as pd
from pathlib import Path
from calculations import coefficients, cross_validate, backwards_selection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
RESULTS = PROJECT_ROOT / "results"


df = pd.read_csv(PROCESSED_DATA)

dummy_cols = df.filter(like="dummy_").columns.tolist()
#save all scores and coeffs in one csv file
metric_rows = []
coef_rows = []


#-------------------------------------------------------------
# linreg with single feature
features = ['is_house', 'accommodates_log', 'accommodates' ,'bedrooms',
            'bathrooms','n_nights', 'days_til_checkin_log', 'days_til_checkin']

# treating the neighbourhood dummies as one variable
variables = {var: [var] for var in features}
variables["neighbourhood"] = dummy_cols


for name, cols in variables.items():
    scores = cross_validate(df, cols)
    metric_rows.append({"model": "single",
                        "variable": name,
                        "n_variables": 1,
                        **scores})      

    coefs = coefficients(df, cols)
    for col, value in coefs.items():
        coef_rows.append({"model": "single",
                          "variable": name,
                          "column": col.replace("dummy_", ""),
                          "coef": value})


#------------------------------------------------------------------
# linear regression with all features

# remove accomodates_log and days_til_checkin, need this for backwards regression later
all_variables = variables.copy()
all_variables.pop("accommodates_log")
all_variables.pop("days_til_checkin")

all_cols = []
for col in all_variables.values():
    all_cols += col

#metrics for linreg on all variables
scores = cross_validate(df, all_cols)
metric_rows.append({"model": "all",
                    "variable": "all",
                    "n_variables": len(all_variables),
                    **scores})

#coeffs for multilinreg model when other vars held fixed
coefs = coefficients(df, all_cols)
for name, cols in all_variables.items():
    for col in cols:
        coef_rows.append({"model" : 'all',
                          "variable": name,
                          "column": col.replace("dummy_", ""),
                          "coef": coefs[col]})



#---------------------------------------------
# backwards linreg
selected, scores, history = backwards_selection(df, all_variables)

print("Backward elimination steps:")
print(history.round(4).to_string(index=False))

# Only the variables that were kept
backward_variables = {}
for name in selected:
    backward_variables[name] = all_variables[name]

backward_cols = []
for cols in backward_variables.values():
    backward_cols += cols

metric_rows.append({"model": "backward",
                    "variable": "selected",
                    "n_variables": len(backward_variables),
                    **scores})

# Coefficients of the selected model (all data)
coefs = coefficients(df, backward_cols)
for name, cols in backward_variables.items():
    for col in cols:
        coef_rows.append({"model": "backward",
                          "variable": name,
                          "column": col.replace("dummy_", ""),
                          "coef": coefs[col]})
        

metrics_df = pd.DataFrame(metric_rows)
coefs_df = pd.DataFrame(coef_rows)

metrics_df.to_csv(RESULTS / "city_linreg_metrics.csv", index=False)
coefs_df.to_csv(RESULTS / "city_linreg_coefs.csv", index=False)