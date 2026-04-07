import pandas as pd

def remove_unused_columns(df: pd.DataFrame):
    print('Removing columns...')
    unused_columns = ['delivery_time_hours', 'expected_time_hours', 'weather_condition',
                      'delayed', 'delivery_status', 'delivery_rating']
    for c in unused_columns:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            print(f'Removed: {c}')



