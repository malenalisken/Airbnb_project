# Airbnb_project
## Project description
Project description
Airbnb has become one of the main platforms tourists and travellers use to book accommodation. It is especially popular with those who want amenities such as a kitchen or a living area, and it is often cheaper than staying at a hotel. However, the platform has a huge number of listings, most of them run by individual hosts, so finding a reasonably priced stay takes a lot of research and comparison. Even then, it can be hard to get a clear picture of what a fair price looks like in a given city.

This project aims to address that problem by building a model that predicts the price per night of an Airbnb listing based on its features. By comparing the predicted price with the listing's actual price, we can assess whether the stay is good value or on the expensive side. At a later stage, a GenAI agent will be added to make the tool easier to use. The agent will collect the necessary information from the user, explain why a price is considered cheap, fair or expensive, and suggest what kind of listings to look for if the user wants a cheaper option.

## Data
Inside Airbnb, Oslo, listings.csv.gz (snapshot: <25.september>) from https://insideairbnb.com/get-the-data/.
Unzip it to data/raw/listings.csv.

## How to run
```bash
pip install -r requirements.txt
python src/cleaning.py
python src/model_city.py
python src/model_district.py
```

EDA: `notebooks/listings.ipynb`

## Contributors
- Niyat Tesfaghabr
- Rohan Uteng 
- Malena Lisken
