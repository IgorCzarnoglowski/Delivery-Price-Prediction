from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def evaluate_model(prediction: dict, y_test: pd.DataFrame):
    # Evaluating models by mean_absolute_error, mean_squared_error, r2_score
    print(f'Evaluating model {prediction['model_name']}')

    mae = mean_absolute_error(y_test, prediction['prediction'])
    mse = mean_squared_error(y_test, prediction['prediction'])
    r2 = r2_score(y_test, prediction['prediction'])

    evaluation_scores = {
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
        'r2_score': r2
    }
    return evaluation_scores


