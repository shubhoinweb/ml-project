from collections import Counter
import kagglehub
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def load_iris_dataset(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    iris = load_iris()
    X = iris.data
    y = iris.target

    feature_summary = []
    for idx, feature_name in enumerate(iris.feature_names):
        feature_values = X[:, idx]
        feature_summary.append(
            {
                "feature": feature_name,
                "count": int(feature_values.size),
                "mean": float(np.mean(feature_values)),
                "std": float(np.std(feature_values)),
                "min": float(np.min(feature_values)),
                "max": float(np.max(feature_values)),
            }
        )

    target_distribution = dict(Counter(y))
    iris.descriptive_stats = {
        "feature_summary": feature_summary,
        "target_distribution": target_distribution,
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, iris
