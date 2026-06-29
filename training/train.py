from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE


def train_model(X_train, y_train, random_state=RANDOM_STATE):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_sample(model, sample):
    prediction = model.predict(sample)
    return prediction[0]
