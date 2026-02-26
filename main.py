import pandas as pd
import numpy as np
from functions import load_csv,merge_datasets,preprocess_data, save_csv, save_to_database
from sqlalchemy import create_engine

# Database connection setup
engine = create_engine("postgresql+psycopg2://username:password@localhost:5432/rapido_db")

# Load the dataset
bookings = load_csv("data/bookings.csv")
drivers = load_csv("data/drivers.csv")
customers = load_csv("data/customers.csv")
location_demand = load_csv("data/location_demand.csv")
time_features = load_csv("data/time_features.csv")
# Merge datasets
df = merge_datasets(bookings, drivers, customers, location_demand, time_features)

# Feature Engineering and Preprocessing
df = preprocess_data(df)

# Save the preprocessed data to a new CSV file and to the database
save_csv(df, "data/preprocessed_rapido_dataset.csv")
save_to_database(df, "preprocessed_rapido_dataset", engine)


print(df.info())
