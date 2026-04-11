from src.data_loader import load_data
from src.data_preprocessing import (
    remove_unused_columns,
    encode_categorical_values,
    create_interaction_features,
    train_test_split_data
)
from src.models import train_models
from src.evaluate import evaluate_model, feature_importance_analysis

def main():
    # 1. Load data
    df = load_data()
    print(f"Data loaded: {df.shape}\n")

    # 2. Preprocess
    remove_unused_columns(df)
    df = encode_categorical_values(df)
    df = create_interaction_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}\n")
    print(X_train.columns.tolist())

    # 4. Train
    predictions = train_models(X_train, y_train, X_test)

    # 5. Evaluate
    results = {}
    for model_name, prediction in predictions.items():
        scores = evaluate_model(prediction, y_test)
        results[model_name] = scores
        print(f"\n{model_name} results:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
        feature_importance_analysis(predictions[model_name], X_test, y_test)

    # 6. Best model by R2
    best_model = max(results, key=lambda x: results[x]['r2_score'])
    print(f"\nBest model: {best_model} (R2: {results[best_model]['r2_score']:.4f})")



if __name__ == '__main__':
    main()