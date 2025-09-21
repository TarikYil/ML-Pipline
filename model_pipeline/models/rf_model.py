# model_pipeline/models/rf_model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_rf_pipeline(n_estimators=200, max_depth=10):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth))
    ])
    return pipeline
