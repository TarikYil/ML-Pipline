# model_pipeline/register.py
import mlflow

def register_model(model_uri, model_name):
    result = mlflow.register_model(model_uri, model_name)
    print(f"Model {model_name} kaydedildi, versiyon: {result.version}")
    return result
