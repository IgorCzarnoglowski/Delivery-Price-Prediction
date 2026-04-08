import pandas as pd
import os

DATASET_DIR = os.path.join(os.getcwd(), '..', 'data')

def load_data():
    df = pd.read_csv(os.path.join(DATASET_DIR, 'Delivery_Logistics.csv'))

    df.rename(columns={'distance_km': 'delivery_distance_km'}, inplace=True)

    return df