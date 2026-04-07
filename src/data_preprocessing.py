import pandas as pd
from sklearn.preprocessing import LabelEncoder

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






