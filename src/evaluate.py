from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

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

def feature_importance_analysis(
    prediction: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    top_n: int = 15
) -> pd.DataFrame | None:
    model_name = prediction["model_name"]
    estimator = prediction["model"]

    print("\n" + "=" * 50)
    print(f"FEATURE IMPORTANCE ANALYSIS — {model_name}")
    print("=" * 50)

    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps["model"]
        feature_names = _get_feature_names_from_pipeline(estimator, X_test)
    else:
        model = estimator
        feature_names = list(X_test.columns)

    # --- Built-in importance (tree-based models) ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp_df = _build_importance_df(feature_names, importances)

    # --- Coefficient-based importance (linear models) ---
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_).flatten()
        feature_imp_df = _build_importance_df(feature_names, importances)

    # --- Permutation importance fallback ---
    else:
        print("No built-in importance found — using permutation importance (slower)...")
        result = permutation_importance(
            estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        feature_imp_df = _build_importance_df(
            list(X_test.columns), result.importances_mean
        )

    # --- Print top N ---
    print(f"\nTop {min(top_n, len(feature_imp_df))} Most Important Features:")
    print(feature_imp_df.head(top_n).to_string(index=False))

    # --- Plot ---
    _plot_importances(feature_imp_df.head(top_n), model_name)

    return feature_imp_df


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_importance_df(feature_names: list, importances) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_names[: len(importances)], "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _get_feature_names_from_pipeline(pipeline, X_test: pd.DataFrame) -> list:
    """Extract feature names after preprocessing step in a Pipeline."""
    if "preprocessor" not in pipeline.named_steps:
        return list(X_test.columns)

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(columns))
        elif hasattr(transformer, "named_steps"):
            # nested pipeline inside ColumnTransformer
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                feature_names.extend(last_step.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
        else:
            feature_names.extend(columns)

    return feature_names


def _plot_importances(df: pd.DataFrame, model_name: str) -> None:
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, y="feature", x="importance", palette="viridis", legend=False)
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


