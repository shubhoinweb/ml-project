import sys
import unittest
from pathlib import Path

from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_FLOWER
from src.data.load_data import load_iris_dataset
from src.models.train_model import predict_sample, train_model


class ModelTrainingTests(unittest.TestCase):
    def test_load_iris_dataset_includes_descriptive_stats(self):
        _, _, _, _, iris = load_iris_dataset()

        self.assertTrue(hasattr(iris, "descriptive_stats"))
        stats = iris.descriptive_stats
        self.assertIn("feature_summary", stats)
        self.assertIn("target_distribution", stats)
        self.assertEqual(len(stats["feature_summary"]), len(iris.feature_names))

    def test_model_training_and_prediction(self):
        X_train, X_test, y_train, y_test, _ = load_iris_dataset()
        model = train_model(X_train, y_train)

        y_pred = model.predict(X_test)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.8)

        prediction = predict_sample(model, SAMPLE_FLOWER)
        self.assertIn(prediction, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
