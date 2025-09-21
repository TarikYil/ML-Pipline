# model_pipeline/train_text.py
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from .models.rf_model import get_rf_pipeline
from .models.lr_model import get_lr_pipeline
from .models.lstm_model import get_lstm_model
from .evaluate import evaluate_regression
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle

def train_model(df, model_type="random_forest", params=None, target_col="target"):
    """
    df: DataFrame
    model_type: "random_forest", "linear_regression" veya "lstm" (zaman serisi)
    params: hiperparametre dict'i
    """
    mlflow.set_experiment("Classic_Models")
    params = params or {}

    # ------------------------------------------------------------------
    # RANDOM FOREST veya LINEAR REGRESSION
    # ------------------------------------------------------------------
    if model_type in ["random_forest","linear_regression"]:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        with mlflow.start_run():
            if model_type=="random_forest":
                model = get_rf_pipeline(
                    n_estimators=params.get("n_estimators",200),
                    max_depth=params.get("max_depth",10)
                )
            else:
                model = get_lr_pipeline()

            model.fit(X_train,y_train)
            preds=model.predict(X_test)
            metrics=evaluate_regression(y_test,preds)

            for k,v in metrics.items():
                mlflow.log_metric(k,v)
            mlflow.sklearn.log_model(model,"model")
            print("Metrikler:",metrics)
            return model

    # ------------------------------------------------------------------
    # LSTM ZAMAN SERİSİ (FORECAST)
    # ------------------------------------------------------------------
    elif model_type=="lstm":
        if target_col not in df.columns:
            raise ValueError(f"DataFrame'de '{target_col}' kolonu yok.")
        series = df[target_col].values.reshape(-1,1)

        # Parametreler
        window_size = params.get("window_size", 50)
        epochs = params.get("epochs", 5)
        batch_size = params.get("batch_size", 32)
        learning_rate = params.get("learning_rate", 0.0001)
        forecast_steps = params.get("forecast_steps", 1)  # yeni parametre

        if len(series) < window_size + 10:
            raise ValueError("Veri sayısı window_size için yetersiz. Daha fazla veri gerekli.")

        # Ölçekleme
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series)

        # X, y dizileri
        X, y = [], []
        for i in range(window_size, len(series_scaled)):
            X.append(series_scaled[i - window_size:i, 0])
            y.append(series_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Train/test split (zaman serisi için son %20 test)
        split_idx = int(len(X)*0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        with mlflow.start_run():
            mlflow.log_param("window_size", window_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("forecast_steps", forecast_steps)

            # Model oluştur
            model = get_lstm_model(window_size, features=1)
            model.optimizer.learning_rate = learning_rate

            # Callback’ler
            callbacks = [
                ModelCheckpoint("best_lstm_model.h5", save_best_only=True, monitor='loss', mode='min', verbose=1),
                EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
            ]

            # Eğit
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

            # Tahmin ve metrikler (test seti)
            preds = model.predict(X_test)
            mse = np.mean((preds.flatten()-y_test)**2)
            mlflow.log_metric("mse",mse)

            # 1️⃣ Forecast: Son pencereyle ileri tahmin
            last_window = series_scaled[-window_size:]  # son window
            forecast = []
            current_input = last_window.copy()
            for _ in range(forecast_steps):
                input_reshaped = current_input.reshape(1, window_size, 1)
                next_pred = model.predict(input_reshaped)[0][0]
                forecast.append(next_pred)
                # Kaydır pencereyi
                current_input = np.append(current_input[1:], next_pred)
            forecast = np.array(forecast).reshape(-1,1)
            forecast_inverse = scaler.inverse_transform(forecast)

            print("🔮 Forecast (ileriye dönük tahmin):", forecast_inverse.flatten())

            # Model ve scaler kaydı
            model.save("lstm_soc_model_final.h5")
            with open("scaler_soc.pkl","wb") as f:
                pickle.dump(scaler,f)

            mlflow.log_artifact("lstm_soc_model_final.h5")
            mlflow.log_artifact("best_lstm_model.h5")
            mlflow.log_artifact("scaler_soc.pkl")

            print("✅ LSTM forecast eğitildi ve MLflow'a kaydedildi. MSE:", mse)
            return model, forecast_inverse.flatten().tolist()

    else:
        raise ValueError("Desteklenmeyen model tipi")
