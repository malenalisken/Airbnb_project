import joblib
import pandas as pd
from pathlib import Path
from trees_func import tuned_gb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(PROCESSED_DATA)

# same features as in the nested cv
features = ['is_house', 'accommodates', 'bedrooms', 'bathrooms',
            'n_nights', 'days_til_checkin', 'latitude', 'longitude']


# tune settings and refit on all data
final_gb = tuned_gb().fit(df[features], df["log_price"])

print("Chosen settings:", final_gb.best_params_)
print("Number of trees:", final_gb.best_estimator_.n_iter_)

# save the fitted model so it can be used without training again
joblib.dump(final_gb.best_estimator_, MODELS / "final_gb.joblib")