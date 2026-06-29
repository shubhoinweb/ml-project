from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_univariate_plots(df, features, output_dir="outputs/figures", show=False):
    """Save univariate plots (histogram, KDE, box, violin) for each feature.

    Args:
        df: pandas DataFrame containing the data.
        features: list of column names to plot.
        output_dir: directory where figures will be saved.
        show: if True, will attempt to show figures (ignored in headless mode).
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import seaborn as sns

        sns.set(style="whitegrid")
    except Exception:
        sns = None

    import pandas as pd
    from scipy.stats import gaussian_kde

    for feat in features:
        if feat not in df.columns:
            # try a case-insensitive match for common variant names
            matches = [c for c in df.columns if c.lower().startswith(feat.lower())]
            if matches:
                feat_name = matches[0]
            else:
                continue
        else:
            feat_name = feat

        series = pd.Series(df[feat_name]).dropna()

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Histogram with KDE overlay
        ax = axes[0, 0]
        ax.hist(series, bins=20, color="steelblue", alpha=0.7)
        ax.set_title(f"Histogram - {feat_name}")
        try:
            xvals = np.linspace(series.min(), series.max(), 200)
            kde = gaussian_kde(series)
            ax2 = ax.twinx()
            ax2.plot(xvals, kde(xvals), color="red")
            ax2.set_ylabel("Density")
        except Exception:
            pass

        # KDE plot
        ax = axes[0, 1]
        if sns is not None:
            sns.kdeplot(series, ax=ax, fill=True, color="purple")
        else:
            try:
                xvals = np.linspace(series.min(), series.max(), 200)
                kde = gaussian_kde(series)
                ax.plot(xvals, kde(xvals), color="purple")
            except Exception:
                ax.plot(series.sort_values().values, color="purple")
        ax.set_title(f"KDE - {feat_name}")

        # Box plot
        ax = axes[1, 0]
        if sns is not None:
            sns.boxplot(x=series, ax=ax, color="lightgreen")
        else:
            ax.boxplot(series)
        ax.set_title(f"Box plot - {feat_name}")

        # Violin plot
        ax = axes[1, 1]
        if sns is not None:
            sns.violinplot(x=series, ax=ax, color="lightblue")
        else:
            ax.hist(series, bins=20, color="lightblue")
        ax.set_title(f"Violin plot - {feat_name}")

        plt.tight_layout()
        out_path = output_dir / f"{feat_name.replace(' ', '_')}_univariate.png"
        fig.savefig(out_path, dpi=200)
        if show:
            try:
                fig.show()
            except Exception:
                pass
        plt.close(fig)


def save_feature_importance_plot(model, feature_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feature_names, importances, color="steelblue")
    ax.set_title("Random Forest Feature Importances")
    ax.set_ylabel("Importance")
    ax.set_ylim(0, max(importances) + 0.1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_iris_scatter_plot(iris, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X = iris.data[:, [0, 2]]
    y = iris.target

    fig, ax = plt.subplots(figsize=(6, 4))
    for target in range(3):
        mask = y == target
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            label=iris.target_names[target],
            alpha=0.7,
        )

    ax.set_xlabel("Sepal length (cm)")
    ax.set_ylabel("Petal length (cm)")
    ax.set_title("Iris dataset")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_plot(y_true, y_pred, class_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_bivariate_plots(df, features=None, hue="species", output_dir="outputs/figures/bivariate", show=False):
    """Generate and save bivariate analysis plots:
    - pairplot (if seaborn available)
    - correlation heatmap
    - scatter plots for each pair colored by `hue` when available

    Args:
        df: pandas DataFrame with data
        features: list of feature column names to include (defaults to all numeric)
        hue: column name for categorical hue (optional)
        output_dir: where to save plots
        show: attempt to show figures
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    try:
        import seaborn as sns
        sns.set(style="ticks")
    except Exception:
        sns = None

    # select features
    if features is None:
        # pick numeric dtypes
        features = list(df.select_dtypes(include="number").columns)

    data = df[features].copy()

    # Correlation heatmap
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    else:
        im = ax.imshow(corr, cmap="coolwarm")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(features)))
        ax.set_yticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_yticklabels(features)
    ax.set_title("Feature Correlation")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    if show:
        try:
            fig.show()
        except Exception:
            pass
    plt.close(fig)

    # Pairplot or scatter matrix
    if sns is not None:
        try:
            pp = sns.pairplot(df[features + ([hue] if hue in df.columns else [])], hue=hue if hue in df.columns else None, corner=False)
            pp.fig.suptitle("Pairplot", y=1.02)
            pp.fig.savefig(output_dir / "pairplot.png", dpi=200)
            plt.close(pp.fig)
        except Exception:
            pass
    else:
        # fallback: scatter matrix
        from pandas.plotting import scatter_matrix

        fig = plt.figure(figsize=(8, 8))
        axes = scatter_matrix(data, diagonal="kde", figsize=(8, 8))
        plt.suptitle("Scatter Matrix")
        fig.savefig(output_dir / "scatter_matrix.png", dpi=200)
        plt.close(fig)

    # Individual pair scatter plots colored by hue if available
    import itertools

    for a, b in itertools.combinations(features, 2):
        fig, ax = plt.subplots(figsize=(6, 4))
        if hue in df.columns and sns is not None:
            sns.scatterplot(data=df, x=a, y=b, hue=hue, ax=ax, palette="deep")
        else:
            ax.scatter(df[a], df[b], alpha=0.7, color="steelblue")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"{a} vs {b}")
        plt.tight_layout()
        fig.savefig(output_dir / f"{a}_vs_{b}.png", dpi=200)
        if show:
            try:
                fig.show()
            except Exception:
                pass
        plt.close(fig)
