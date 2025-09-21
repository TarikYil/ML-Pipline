# model_pipeline/models/lr_model.py
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_lr_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    return pipeline
