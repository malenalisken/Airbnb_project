"""Prediction service for the project's existing final gradient boosting model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['is_house', 'accommodates', 'bedrooms', 'bathrooms',
            'n_nights', 'days_til_checkin', 'latitude', 'longitude']


def areas():
    """Areas actually represented in the cleaned project dataset."""
    df = pd.read_csv(ROOT / 'data/processed/listings_clean.csv',
                     usecols=['neighbourhood_cleansed', 'latitude', 'longitude'])
    df = df.dropna()
    df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].str.replace('_', ' ', regex=False)
    return df


def predict_price(area: str, is_house: bool, accommodates: int, bedrooms: int,
                  bathrooms: float, n_nights: int, days_til_checkin: int,
                  latitude: float | None = None, longitude: float | None = None) -> dict:
    """Predict nightly price with the group's existing model.

    Args:
      area: Oslo district name as represented in the processed dataset.
      is_house: True for house, False for apartment.
      accommodates: Maximum number of guests (1-12).
      bedrooms: Number of bedrooms (0 or more).
      bathrooms: Number of bathrooms (greater than 0, halves permitted).
      n_nights: Length of stay (1-28).
      days_til_checkin: Days between booking/quote date and check-in (0 or more).
      latitude: Exact latitude if known; otherwise use area median.
      longitude: Exact longitude if known; otherwise use area median.
    """
    if not isinstance(is_house, bool):
        return {'error': 'is_house must be true or false'}
    if not (1 <= accommodates <= 12 and bedrooms >= 0 and bathrooms > 0 and 1 <= n_nights <= 28 and days_til_checkin >= 0):
        return {'error': 'Invalid housing or stay values: guests 1-12, bedrooms >=0, bathrooms >0, nights 1-28, days to check-in >=0'}
    if (latitude is None) != (longitude is None):
        return {'error': 'Provide both latitude and longitude, or neither.'}
    df = areas()
    names = sorted(df['neighbourhood_cleansed'].unique().tolist())
    matched = next((name for name in names if name.casefold() == area.strip().replace('_', ' ').casefold()), None)
    if matched is None:
        return {'error': f'Unknown area: {area}', 'available_areas': names}
    if latitude is None:
        subset = df[df['neighbourhood_cleansed'] == matched]
        latitude = float(subset['latitude'].median())
        longitude = float(subset['longitude'].median())
        position_source = 'area_median_proxy'
    else:
        position_source = 'user_coordinates'
    if not (np.isfinite(latitude) and np.isfinite(longitude) and -90 <= latitude <= 90 and -180 <= longitude <= 180):
        return {'error': 'Invalid coordinates'}
    row = pd.DataFrame([{'is_house': int(is_house), 'accommodates': accommodates,
                         'bedrooms': bedrooms, 'bathrooms': bathrooms,
                         'n_nights': n_nights, 'days_til_checkin': days_til_checkin,
                         'latitude': latitude, 'longitude': longitude}], columns=FEATURES)
    model = joblib.load(ROOT / 'models/final_gb.joblib')
    estimate = float(np.exp(model.predict(row)[0]))
    return {'estimated_price_per_night': round(estimate, 2),
            'area': matched, 'latitude': latitude, 'longitude': longitude,
            'position_source': position_source,
            'currency': 'verify_against_dataset',
            'note': 'Model estimate, not guaranteed market price. Area median is a geographic proxy, not an actual listing address.'}
