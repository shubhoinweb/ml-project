# Iris Random Forest

This project trains and evaluates a Random Forest classifier on the Iris dataset.

## Run the project

From the project root, run:

```bash
python main.py
```

## Project structure

 - `main.py`: entry point for training and prediction
 - `preprocessing/preprocessing.py`: loads and splits the dataset
 - `training/train.py`: trains the model and predicts samples
 - `01_eda/eda.py` (now `eda/eda.py`): exploratory data analysis utilities
 - `evaluation/visuals.py`: plotting and evaluation helpers
 - `tests/test_model.py`: basic verification test

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run the project from the repository root:

```bash
python main.py
```
