"""
Cleans the raw Inside Airbnb listings data for price prediction.

Input:  data/raw/listings.csv (download instructions in the README)
Output: data/processed/listings_clean.csv, indexed by listing id

Steps:
1. Keep the columns used for modelling and set listing id as the index
   (stops if any id appears more than once).
2. Convert price to a number, drop listings with missing or zero price,
   and remove the top 1% of prices as outliers.
3. Keep only entire homes and apartments, and encode property type as
   is_house (1 = house, 0 = apartment).
4. Clean the remaining features: superhost as 0/1, bathrooms as a number,
   and remove listings with 13 guests or a minimum stay over 28 nights.
5. Create has_rating and a weighted (Bayesian-average) rating, so listings
   with few reviews are pulled toward the median rating.
6. Check that no missing values remain, add log-transformed versions of
   price, accommodates and minimum_nights.
7. Group Stovner/Grorud/Alna, drop Marka, add neighbourhood dummies
   (dummy_<name>, Sentrum as baseline) and save the result.

Run from anywhere with:  python src/cleaning.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "listings.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
df = pd.read_csv(RAW_DATA)


def basic_clean(df):
    features_to_keep = ['last_scraped', 
                        'neighbourhood_cleansed', 
                        'latitude', 
                        'longitude',
                        'property_type', 
                        'accommodates', 
                        'bathrooms_text', 
                        'bedrooms', 
                        'beds',
                        'price_quote_checkin_date', 
                        'price_quote_checkout_date', 
                        'price_quote_price_per_night']
    df_bc = df[features_to_keep].copy()

    # removing outliers in price and listings where price is nan
    cutoff = df_bc['price_quote_price_per_night'].quantile(0.99)
    df_bc = df_bc.loc[df_bc['price_quote_price_per_night'] <= cutoff]
    df_bc = df_bc.loc[df_bc['price_quote_price_per_night'].notna()]
    df_bc = df_bc.rename(columns = {'price_quote_price_per_night' : 'price_per_night'})

    # Property_type : keep property types that are relativly standard for city
    entire_space = [
        "Entire rental unit", "Entire condo", "Entire home", "Entire townhouse",
        "Entire loft", "Entire villa", "Entire serviced apartment", "Entire guest suite",
        "Entire guesthouse", "Tiny home", "Entire cabin",
    ]
    df_bc = df_bc[df_bc["property_type"].isin(entire_space)]
    #replace categories with home = 1 apt = 0 
    homes = [
        "Entire home", "Entire townhouse", "Entire villa",
        "Entire guesthouse", "Tiny home", "Entire cabin",
    ]
    df_bc["property_type"] = df_bc["property_type"].isin(homes).astype(int)
    df_bc = df_bc.rename(columns={"property_type": "is_house"})

    # Bathrooms: take out the number from text and remove listings with 0 bathrooms
    df_bc["bathrooms"] = df_bc["bathrooms_text"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
    df_bc.loc[df_bc["bathrooms_text"] == "Half-bath", "bathrooms"] = 0.5
    df_bc = df_bc[df_bc["bathrooms"] > 0]
    df_bc = df_bc.drop(columns=["bathrooms_text"])

    # Nights: create a column with number of nights base on checkin and checout and cut off long term stays
    df_bc['price_quote_checkin_date'] = pd.to_datetime(df_bc['price_quote_checkin_date'])
    df_bc['price_quote_checkout_date'] = pd.to_datetime(df_bc['price_quote_checkout_date'])

    df_bc['n_nights'] = (df_bc['price_quote_checkout_date'] - df_bc['price_quote_checkin_date']).dt.days
    df_bc = df_bc[df_bc['n_nights'] <= 28]

    # Days until checkin:
    df_bc['last_scraped'] = pd.to_datetime(df_bc['last_scraped'])
    df_bc['days_til_checkin'] = (df_bc['price_quote_checkin_date'] - df_bc['last_scraped']).dt.days
    df_bc = df_bc.drop(columns = ['last_scraped'])
    
    #Checkin month based on checkin date
    df_bc['checkin_month'] = df_bc['price_quote_checkin_date'].dt.month

    df_bc = df_bc.drop(columns = ['price_quote_checkout_date', 'price_quote_checkin_date'])

    return df_bc

def final_clean(df):
    features_to_keep = {'log_price', 
                        'price_per_night', 
                        'accomodates', 
                        'bedrooms', 
                        'bathrooms', 
                        'is_house', 
                        'days_til_checkin', 
                        'neighbourhood_cleansed', 
                        'latitude', 
                        'longitude', 
                        'n_nights'}
    
    df_fc = df[features_to_keep].copy()

    #cutoff for listings with more than 12 accomodations
    df_fc = df_fc[df_fc["accommodates"] <= 12]

    # Neighbourhoods: remove spacing in values 
    df_fc["neighbourhood_cleansed"] = (df_fc["neighbourhood_cleansed"].str.strip().str.replace(" ", "_"))
    df_fc['neighbourhood_grouped'] = df_fc['neighbourhood_cleansed'].replace({
        "Stovner": "Stovner/Grorud/Alna",
        "Grorud": "Stovner/Grorud/Alna",
        "Alna": "Stovner/Grorud/Alna",
    })
    df_fc = df_fc[df_fc['neighbourhood_cleansed'] != 'Marka'].copy()

    # Neighbourhood dummies: one column per grouped neighbourhood, Sentrum = baseline (no column)
    dummies = pd.get_dummies(df_fc['neighbourhood_grouped'], prefix="dummy", dtype=int)
    dummies = dummies.drop(columns="dummy_Sentrum")
    df_fc = pd.concat([df_fc, dummies], axis=1)

    #transformation of accomodates, days until checkin and price

    #transformation of accomodates, days until checkin and price
    df_fc['log_price'] = np.log(df_fc['price_per_night'])
    df_fc['accommodates_log'] = np.log(df_fc['accommodates'])
    # log1p = log(1 + x), since some listings have 0 days until checkin
    df_fc['days_til_checkin_log'] = np.log1p(df_fc['days_til_checkin'])

    return df_fc








