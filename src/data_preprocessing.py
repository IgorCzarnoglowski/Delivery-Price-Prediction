import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def remove_unused_columns(df: pd.DataFrame):
    print('Removing columns...')
    unused_columns = ['delivery_time_hours', 'expected_time_hours', 'weather_condition',
                      'delayed', 'delivery_status', 'delivery_rating']
    for c in unused_columns:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            print(f'Removed: {c}')

def encode_categorical_values(df: pd.DataFrame):
    print('Encoding categorical features...')

    categorical_columns= df.select_dtypes(include='object').columns

    for col in categorical_columns:
        # One hot encoding for less than 6 features
        if df[col].nunique() <= 6:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
        # Label encoding for more than 6
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df

def create_interaction_features(df: pd.DataFrame):
    # Create interaction features between important variables
    print('Creating interaction features...')

    if 'delivery_distance_km' in df.columns and 'package_weight_kg' in df.columns:
        df['distance_weight_interaction'] = df['delivery_distance_km'] * df['package_weight_kg']

        print('Create distance_weight_interaction')


def create_polynomial_features(df: pd.DataFrame, degree = 2):
    # Create polynomial features for important numerical variables

    print('Creating polynomial features...')

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    if 'delivery_cost' in df.columns and len(numerical_cols) > 0:
        try:
            corr_with_target = df[numerical_cols.tolist()].corr()[
                'delivery_cost'].abs().sort_values(ascending=False)
            top_features = corr_with_target[1:4].index.tolist()  # Top 3 features excluding target itself

            for feature in top_features:
                for deg in range(2, degree + 1):
                    df[f'{feature}_power_{deg}'] = df[feature] ** deg
                    print(f"Created {feature}_power_{deg}")
        except Exception as e:
            # Fallback: use first few numerical features
            for feature in numerical_cols[:2]:
                for deg in range(2, degree + 1):
                    df[f'{feature}_power_{deg}'] = df[feature] ** deg
                    print(f"Created {feature}_power_{deg}")

    return df








