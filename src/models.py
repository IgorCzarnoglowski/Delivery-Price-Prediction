from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV


def initialize_models():
    # Initializing multiple models
    print('Initilizing models...')

    models = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, verbosity=0),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 6, 9],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42, verbose=-1),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [5, 10, -1],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__num_leaves': [31, 50, 100]
            }
        }
    }

    return models

def train_models(X_train, y_train, X_test, cv: int = 5):
    # Training models using hyperparameter tuning
    print('Training models...\n')

    models_config = initialize_models()
    models = defaultdict(dict)
    predictions = defaultdict(dict)


    for name, config in models_config.items():
        print(f"Training {name}...")

        grid_search = GridSearchCV(
            estimator = config['model'],
            param_grid = config['params'],
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        models[name] = {
            'grid_search': grid_search,
            'best_estimator': grid_search.best_estimator_
        }

        prediction = grid_search.best_estimator_.predict(X_test)
        predictions[name] = {
            'model_name': name,
            'prediction': prediction
        }

    return predictions



