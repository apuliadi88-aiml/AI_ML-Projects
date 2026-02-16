import pandas as pd
import numpy as np


def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"CSV loaded successfully: {file_path} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}")
        exit(1) #Failed due to an error
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)
    

def merge_datasets(df_bookings, df_drivers, df_customers, df_location_demand, df_time_features):
    df_customers = df_customers.rename(columns={
        "completed_rides": "customer_completed_rides",
        "cancelled_rides": "customer_cancelled_rides",
        "incomplete_rides": "customer_incomplete_rides"
        })
    df_drivers = df_drivers.rename(columns={"vehicle_type": "driver_vehicle_type"})
    df_location_demand = df_location_demand.rename(columns={"vehicle_type": "location_vehicle_type"})


    df = df_bookings.merge(df_customers, on="customer_id", how="left")
    df = df.merge(df_drivers, on="driver_id", how="left")
    df = df.merge(
        df_location_demand,
        left_on=["city", "pickup_location", "hour_of_day", "vehicle_type"],
        right_on=["city", "pickup_location", "hour_of_day", "location_vehicle_type"],
        how="left"
    )
    
    return df

def preprocess_data(df):
    df['peak_hour_flag'] = (df['hour_of_day'].isin([7,8,9,10,17,18,19,20])).astype(int)

    # Fill the missing values in 'Incomplete Ride Reason' as 'Not Applicable'
    df['incomplete_ride_reason'].value_counts(dropna=False)
    df['incomplete_ride_reason']= df['incomplete_ride_reason'].fillna('Not Applicable')

    # Fill missing actual ride time with estimated ride time
    df['actual_ride_time_min'] = df['actual_ride_time_min'].fillna(0.0)

    # fare_per_km
    df['fare_per_km'] = df['booking_value'] / df['ride_distance_km'].replace(0, np.nan)
    df['fare_per_km'] = df['fare_per_km'].fillna(0)

    # fare_per_min
    df['fare_per_min'] = df['booking_value'] / df['estimated_ride_time_min'].replace(0, np.nan)
    df['fare_per_min'] = df['fare_per_min'].fillna(0)

    # long_distance_flag
    threshold = df['ride_distance_km'].quantile(0.75)  # top 25% as long distance
    df['long_distance_flag'] = (df['ride_distance_km'] > threshold).astype(int)

    # city_pair
    df['city_pair'] = df['pickup_location'] + "_" + df['drop_location']

    # driver_reliability_score
    df['incomplete_ratio'] = df['incomplete_rides'] / df['total_assigned_rides'].replace(0,1)
    df['rating_norm'] = df['avg_driver_rating'] / 5
    df['pickup_delay_norm'] = 1 - (df['avg_pickup_delay_min'] / 30).clip(0,1)

    df['driver_reliability_score'] = (
        0.30 * df['acceptance_rate'] +
        0.25 * df['rating_norm'] +
        0.20 * (1 - df['delay_rate']) +
        0.15 * df['pickup_delay_norm'] +
        0.10 * (1 - df['incomplete_ratio'])
    ) * 100
    df['driver_reliability_score'] = df['driver_reliability_score'].clip(0, 100)
    # customer_loyalty_score
    df['completion_ratio'] = df['customer_completed_rides'] / df['total_bookings'].replace(0,1)

    df['customer_loyalty_score'] = (
        0.50 * df['completion_ratio'] +
        0.30 * (1 - df['cancellation_rate']) +
        0.20 * (df['avg_customer_rating'] / 5)
    ) * 100
    df['customer_loyalty_score'] = df['customer_loyalty_score'].clip(0, 100)

    df['booking_datetime'] = pd.to_datetime(df['booking_date'] + ' ' + df['booking_time'])

    cols_to_round = [
    'cancellation_rate',
    'avg_wait_time_min',
    'avg_surge_multiplier',
    'fare_per_km',
    'fare_per_min',
    'driver_reliability_score',
    'customer_loyalty_score'
    ]

    # Round to 2 decimal places
    df[cols_to_round] = df[cols_to_round].round(2)

    drop_cols = [
        'incomplete_ratio',
        'rating_norm',
        'delay_rate',
        'pickup_delay_norm',
        'completion_ratio',
        'location_vehicle_type',
        'incomplete_ratio',
        'total_bookings',
        'customer_completed_rides',
        'customer_cancelled_rides',
        'customer_incomplete_rides',
        'total_assigned_rides',
        'accepted_rides',
        'incomplete_rides',
        'booking_date',
        'booking_time',
        'driver_vehicle_type',
        'customer_gender',
        'customer_age',
        'customer_city',
        'customer_signup_days_ago'
        ]
    df_clean = df.drop(columns=drop_cols, errors='ignore')

    cols_to_round = [
    'cancellation_rate',
    'avg_wait_time_min',
    'avg_surge_multiplier',
    'fare_per_km',
    'fare_per_min',
    'driver_reliability_score',
    'customer_loyalty_score'
]

    # Round to 2 decimal places
    df[cols_to_round] = df[cols_to_round].round(2)
    
    return df_clean

def save_to_database(df, table_name,engine):
    try:
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        print(f"Data saved to database successfully: {table_name}")
    except Exception as e:
        print(f"Error saving to database: {e}")

def save_csv(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        print(f"Preprocessed CSV saved: {file_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")