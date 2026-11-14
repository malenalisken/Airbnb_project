
import numpy as np
import pandas as pd



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
    features_to_keep = ['price_per_night', 
                        'accommodates', 
                        'bedrooms', 
                        'bathrooms', 
                        'is_house', 
                        'days_til_checkin', 
                        'neighbourhood_cleansed', 
                        'latitude', 
                        'longitude', 
                        'n_nights']
    
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
    df_fc['log_price'] = np.log(df_fc['price_per_night'])
    df_fc['accommodates_log'] = np.log(df_fc['accommodates'])
    # log1p = log(1 + x), since some listings have 0 days until checkin
    df_fc['days_til_checkin_log'] = np.log1p(df_fc['days_til_checkin'])

    return df_fc


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DATA = PROJECT_ROOT / "data" / "raw" / "listings.csv"
    PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"

    df = pd.read_csv(RAW_DATA)
    df_clean = final_clean(basic_clean(df))
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA, index=False)
    print(df_clean.columns.to_list())








