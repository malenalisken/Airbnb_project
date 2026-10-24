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
   price, accommodates and minimum_nights, and save the result.

Run from anywhere with:  python src/cleaning.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "listings.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
df = pd.read_csv(RAW_DATA)


#deciding which columns to keep for cleaning
wanted_cols = [
    "id",
    "host_is_superhost",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
]

df_clean = df[wanted_cols].copy()

# Id: Each listing should appear once in a single snapshot
if not df_clean["id"].is_unique:
    raise ValueError("Duplicate listing IDs found. Is listings.csv.gz a single snapshot?")

# Use id as the index: not a model feature, but still traceable
df_clean = df_clean.set_index("id")


# Price: Start by cleaning the target column values to float values
df_clean["price"] = df_clean["price"].str.replace("$", "", regex=False)
df_clean["price"] = df_clean["price"].str.replace(",", "", regex=False)
df_clean["price"] = df_clean["price"].str.replace(" ", "", regex=False)
df_clean["price"] = df_clean["price"].astype(float)

# Remove rows with no price or 0
df_clean = df_clean.dropna(subset=["price"])
df_clean = df_clean[df_clean["price"] > 0]
# Remove outliers
cutoff = df_clean["price"].quantile(0.99)
df_clean = df_clean[df_clean["price"] <= cutoff]


# Host_is_superhost : make values binary, 1 if is superhost
df_clean["host_is_superhost"] = df_clean["host_is_superhost"].map({"t": 1, "f": 0})

# Property_type : keep property types that are relativly standard for city
entire_space = [
    "Entire rental unit", "Entire condo", "Entire home", "Entire townhouse",
    "Entire loft", "Entire villa", "Entire serviced apartment", "Entire guest suite",
    "Entire guesthouse", "Tiny home", "Entire cabin",
]
df_clean = df_clean[df_clean["property_type"].isin(entire_space)]

#replace categories with home = 1 apt = 0 
homes = [
    "Entire home", "Entire townhouse", "Entire villa",
    "Entire guesthouse", "Tiny home", "Entire cabin",
]
df_clean["property_type"] = df_clean["property_type"].isin(homes).astype(int)
df_clean = df_clean.rename(columns={"property_type": "is_house"})

# Room_type: should only be entire home/apt given cleaning of property type
if not df_clean["room_type"].eq("Entire home/apt").all():
    raise ValueError("room_type contains values other than 'Entire home/apt'")
df_clean = df_clean.drop(columns = ['room_type'])

# Bathrooms: take out the number from text and remove listings with 0 bathrooms
df_clean["bathrooms"] = df_clean["bathrooms_text"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
df_clean.loc[df_clean["bathrooms_text"] == "Half-bath", "bathrooms"] = 0.5
df_clean = df_clean[df_clean["bathrooms"] > 0]

df_clean = df_clean.drop(columns=["bathrooms_text"])

# Outlier: remove listing with 13 accommodates
df_clean = df_clean[df_clean["accommodates"] != 13]

# Minimum_nights: remove all listings that require a long-time stay
df_clean = df_clean[df_clean["minimum_nights"] <= 28]

# If has a rating: map to 1 if it has rating and 0 if not
df_clean["has_rating"] = df_clean["review_scores_rating"].notna().astype(int)

#-----------------------------------------------------------------------
#.......................................................................
# Might want to do this in the kfold instead

# Review_score: do a weighted rating with bayesian-average , replace no rating with the median
m = df_clean["number_of_reviews"].median()  
if m == 0:
    raise ValueError("Median number_of_reviews is 0; choose m manually")
                               
C = df_clean["review_scores_rating"].median()  
v = df_clean["number_of_reviews"]
R = df_clean["review_scores_rating"]

df_clean["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C
# Listings with no reviews get the typical rating
df_clean["weighted_rating"] = df_clean["weighted_rating"].fillna(C)
df_clean = df_clean.drop(columns=["review_scores_rating"])

#.....................................................................
#---------------------------------------------------------------------

#check if any NaN values in the dataframe
nan_counts = df_clean.isna().sum()
if nan_counts.any():
    raise ValueError(f"df_clean contains NaN values:\n{nan_counts[nan_counts > 0]}")

# Tranformation of target column, accommodates and minimum_nights
df_clean["log_price"] = np.log(df_clean["price"])
df_clean["accommodates_log"] = np.log(df_clean["accommodates"])
df_clean["minimum_nights_log"] = np.log(df_clean["minimum_nights"])

PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(PROCESSED_DATA)
