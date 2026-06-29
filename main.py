import sys
from pathlib import Path

from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eda import print_eda_summary, summarize_dataset
from preprocessing import load_iris_dataset
from training import predict_sample, train_model
from evaluation import (
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_iris_scatter_plot,
)

import pandas as pd
import matplotlib.pyplot as plt
def main():
    X_train, X_test, y_train, y_test, iris = load_iris_dataset(test_size=0.2, random_state=42)
    _, summary = summarize_dataset(iris)
    print_eda_summary(summary)

    print("\nDescriptive Statistics")
    print("-" * 40)
    for feature_stat in iris.descriptive_stats["feature_summary"]:
        print(
            f"{feature_stat['feature']}: mean={feature_stat['mean']:.3f}, "
            f"std={feature_stat['std']:.3f}, min={feature_stat['min']:.3f}, "
            f"max={feature_stat['max']:.3f}"
        )
    print("Target distribution:")
    for target, count in iris.descriptive_stats["target_distribution"].items():
        print(f"  - class {target}: {count}")

    model = train_model(X_train, y_train, random_state=42)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = predict_sample(model, sample)
    print(f"Predicted Class: {iris.target_names[prediction]}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    save_feature_importance_plot(model, iris.feature_names, output_dir / "feature_importance.png")
    save_iris_scatter_plot(iris, output_dir / "iris_scatter.png")
    save_confusion_matrix_plot(y_test, y_pred, iris.target_names, output_dir / "confusion_matrix.png")
    print(f"Saved visuals to {output_dir}")


if __name__ == "__main__":
    main()
