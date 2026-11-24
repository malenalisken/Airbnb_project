
"""Prediction service for the project's existing final gradient boosting model.

The user must explicitly provide days_til_checkin.
Never assume or invent this value.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    'is_house',
    'accommodates',
    'bedrooms',
    'bathrooms',
    'n_nights',
    'days_til_checkin',
    'latitude',
    'longitude'
]


def areas():
    """Read Oslo district coordinates from the project CSV."""

    df = pd.read_csv(
        ROOT / 'data/processed/bydel_koordinater.csv'
    )

    df = df.dropna()

    df['neighbourhood_cleansed'] = (
        df['neighbourhood_cleansed']
        .str.replace('_', ' ', regex=False)
    )

    return df


def predict_price(
    area: str,
    is_house: bool,
    accommodates: int,
    bedrooms: int,
    bathrooms: float,
    n_nights: int,
    days_til_checkin: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None
) -> dict:
    """
    Predict nightly Airbnb price using the trained Gradient Boosting model.

    days_til_checkin must be explicitly provided by the user.
    """

    # Check that check-in timing is provided
    if days_til_checkin is None:
        return {
            'error': (
                'Missing days_til_checkin. '
                'Ask the user how many days remain until check-in '
                'before making a prediction.'
            )
        }

    # Validate input values
    if not isinstance(is_house, bool):
        return {'error': 'is_house must be true or false'}

    if not isinstance(days_til_checkin, int) or days_til_checkin < 0:
        return {
            'error': 'days_til_checkin must be a non-negative integer.'
        }

    if not (
        1 <= accommodates <= 12
        and bedrooms >= 0
        and bathrooms > 0
        and 1 <= n_nights <= 28
    ):
        return {
            'error': (
                'Invalid housing or stay values: '
                'guests 1-12, bedrooms >= 0, bathrooms > 0, '
                'nights 1-28.'
            )
        }

    if (latitude is None) != (longitude is None):
        return {
            'error': 'Provide both latitude and longitude, or neither.'
        }

    # Find neighbourhood coordinates
    df = areas()

    names = sorted(
        df['neighbourhood_cleansed'].unique().tolist()
    )

    matched = next(
        (
            name for name in names
            if name.casefold() == area.strip().replace(
                '_', ' '
            ).casefold()
        ),
        None
    )

    if matched is None:
        return {
            'error': f'Unknown area: {area}',
            'available_areas': names
        }

    # Use neighbourhood median coordinates if exact coordinates
    # are not provided
    if latitude is None:

        subset = df[
            df['neighbourhood_cleansed'] == matched
        ]

        latitude = float(subset['latitude'].median())
        longitude = float(subset['longitude'].median())

        position_source = 'area_median_proxy'

    else:
        position_source = 'user_coordinates'

    # Validate coordinates
    if not (
        np.isfinite(latitude)
        and np.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        return {'error': 'Invalid coordinates'}

    # Prepare model input
    row = pd.DataFrame(
        [{
            'is_house': int(is_house),
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'n_nights': n_nights,
            'days_til_checkin': days_til_checkin,
            'latitude': latitude,
            'longitude': longitude
        }],
        columns=FEATURES
    )

    # Load the existing trained model
    model = joblib.load(ROOT / 'final_gb.joblib')

    # Predict price and convert from log-price
    estimate = float(
        np.exp(model.predict(row)[0])
    )

    return {
        'estimated_price_per_night': round(estimate, 2),
        'area': matched,
        'latitude': latitude,
        'longitude': longitude,
        'position_source': position_source,
        'currency': 'NOK',
        'note': (
            'Model estimate, not guaranteed market price. '
            'Area median is a geographic proxy, '
            'not an actual listing address.'
        )
    }
