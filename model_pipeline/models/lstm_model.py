# model_pipeline/models/lstm_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def get_lstm_model(window_size, features=1):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, features)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mean_squared_error')
    return model
