
"""Prediction service for the project's existing final gradient boosting model."""

from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo
import re

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


def calculate_days_until_checkin(checkin_date: str) -> int:
    """Calculate days until check-in using Norwegian local date."""

    today = datetime.now(ZoneInfo("Europe/Oslo")).date()

    months = {
        "januar": 1,
        "februar": 2,
        "mars": 3,
        "april": 4,
        "mai": 5,
        "juni": 6,
        "juli": 7,
        "august": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "desember": 12
    }

    value = checkin_date.lower().strip()

    # Accept ISO format: YYYY-MM-DD
    try:
        parsed = date.fromisoformat(value)

    except ValueError:
        parsed = None

    if parsed is None:

        # Accept dates such as:
        # 10. desember
        # 10.12
        # 10.12.2026

        match = re.fullmatch(
            r"(\d{1,2})\.?\s*([a-zæøå]+|\d{1,2})\.?"
            r"(?:\s*(\d{4}))?",
            value
        )

        if not match:
            raise ValueError(
                "Invalid check-in date. "
                "Use a date such as 10. desember or 10.12.2026."
            )

        day = int(match.group(1))
        month_text = match.group(2)
        year_text = match.group(3)

        if month_text.isdigit():
            month = int(month_text)
        else:
            month = months.get(month_text)

        if month is None or not 1 <= month <= 12:
            raise ValueError("Invalid month.")

        year = int(year_text) if year_text else today.year

        try:
            parsed = date(year, month, day)
        except ValueError:
            raise ValueError("Invalid calendar date.")

        # If no year was specified, use the next occurrence.
        if year_text is None and parsed < today:
            parsed = date(year + 1, month, day)

    if parsed < today:
        raise ValueError("Check-in date is in the past.")

    return (parsed - today).days


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
    accommodates: int,
    bedrooms: int,
    n_nights: int,
    checkin_date: str,
    is_house: bool | None = None,
    bathrooms: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None
) -> dict:
    """
    Predict nightly Airbnb price using the trained
    Gradient Boosting model.

    The check-in date is provided by the user.
    Python calculates days until check-in.
    """

    # Validate primary information

    if not isinstance(area, str) or not area.strip():
        return {'error': 'A valid Oslo neighbourhood is required.'}

    if not isinstance(accommodates, int) or isinstance(accommodates, bool):
        return {'error': 'Number of guests must be an integer.'}

    if not 1 <= accommodates <= 12:
        return {'error': 'Number of guests must be between 1 and 12.'}

    if not isinstance(bedrooms, int) or isinstance(bedrooms, bool) or bedrooms < 0:
        return {'error': 'Bedrooms must be a non-negative integer.'}

    if not isinstance(n_nights, int) or isinstance(n_nights, bool):
        return {'error': 'Number of nights must be an integer.'}

    if not 1 <= n_nights <= 28:
        return {'error': 'Number of nights must be between 1 and 28.'}

    # Calculate days until check-in in Python

    if not isinstance(checkin_date, str) or not checkin_date.strip():
        return {'error': 'A check-in date is required.'}

    try:
        days_til_checkin = calculate_days_until_checkin(checkin_date)
    except ValueError as error:
        return {'error': str(error)}

    # Apply reasonable defaults for secondary information

    assumptions = []

    if is_house is None:
        if accommodates >= 6:
            return {
                'error': (
                    'Please clarify whether you prefer a house '
                    'or an apartment for this group.'
                )
            }

        is_house = False
        assumptions.append(
            'Apartment assumed because no property type was specified.'
        )

    if not isinstance(is_house, bool):
        return {'error': 'is_house must be true or false.'}

    if bathrooms is None:
        if accommodates >= 4:
            return {
                'error': (
                    'Please specify the preferred number of bathrooms '
                    'for a group of four or more guests.'
                )
            }

        bathrooms = 1.0
        assumptions.append(
            'One bathroom assumed for a group of 1-3 guests.'
        )

    if (
        not isinstance(bathrooms, (int, float))
        or isinstance(bathrooms, bool)
        or not np.isfinite(bathrooms)
        or bathrooms <= 0
    ):
        return {'error': 'Bathrooms must be a positive number.'}

    # Check coordinates

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
        return {'error': 'Invalid coordinates.'}

    # Prepare model input

    row = pd.DataFrame(
        [{
            'is_house': int(is_house),
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'bathrooms': float(bathrooms),
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

    total_price = estimate * n_nights

    return {
        'estimated_price_per_night': round(estimate, 2),
        'estimated_total_price': round(total_price, 2),
        'area': matched,
        'accommodates': accommodates,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'is_house': is_house,
        'n_nights': n_nights,
        'checkin_date': checkin_date,
        'days_til_checkin': days_til_checkin,
        'latitude': latitude,
        'longitude': longitude,
        'position_source': position_source,
        'assumptions': assumptions,
        'currency': 'NOK',
        'note': (
            'Model estimate, not guaranteed market price. '
            'Area median is a geographic proxy, '
            'not an actual listing address.'
        )
    }

