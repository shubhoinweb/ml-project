from pathlib import Path

from training import predict_sample


def predict(model, sample):
    return predict_sample(model, sample)
