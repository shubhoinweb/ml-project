from pathlib import Path

from .visuals import save_confusion_matrix_plot, save_feature_importance_plot, save_iris_scatter_plot


def evaluate_model(model, iris, y_test, y_pred, output_dir=Path('outputs')):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_feature_importance_plot(model, iris.feature_names, output_dir / "feature_importance.png")
    save_iris_scatter_plot(iris, output_dir / "iris_scatter.png")
    save_confusion_matrix_plot(y_test, y_pred, iris.target_names, output_dir / "confusion_matrix.png")
