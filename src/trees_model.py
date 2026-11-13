import pandas as pd
from pathlib import Path
from trees_func import tuned_rf, nested_cv, tuned_gb
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
RESULTS = PROJECT_ROOT / "results"

df = pd.read_csv(PROCESSED_DATA)
metric_rows = []
importance_tables = []

# use different features for random forest as is no longer reliant on linear relationshipd
features = ['is_house', 'accommodates', 'bedrooms', 'bathrooms',
            'n_nights', 'days_til_checkin', 'latitude', 'longitude']


# random foreest
scores, settings, importance = nested_cv(df, tuned_rf(), features)
 
print("Random forest settings chosen per outer fold:")
print(settings.to_string())
 
metric_rows.append({"model": "rf",
                    "n_features": len(features),
                    **scores})

importance.insert(0, "model", "rf")                                     
importance_tables.append(importance) 
 

#-----------------------
# gradient boosting
scores, settings, importance = nested_cv(df, tuned_gb(), features)
 
print("Gradient boosting settings chosen per outer fold:")
print(settings.to_string())
 
metric_rows.append({"model": "gb",
                    "n_features": len(features),
                    **scores})

importance.insert(0, "model", "gb")                                      
importance_tables.append(importance) 

metrics_df = pd.DataFrame(metric_rows)
 
metrics_df.to_csv(RESULTS / "city_trees_metrics.csv", index=False)

importance_df = pd.concat(importance_tables)                             
importance_df.to_csv(RESULTS / "city_trees_importance.csv", index=False)