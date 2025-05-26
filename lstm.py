import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def loadAndPreprocess(csv_path, windowSize=30):
    d = pd.read_csv(csv_path, parse_dates=['timestamp'])
    d.sort_values('timestamp', inplace=True)
    values = d['value'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaledValues = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(scaledValues) - windowSize):
        X.append(scaledValues[i:i+windowSize])
        y.append(scaledValues[i+windowSize])
    
    X, y = np.array(X), np.array(y)
    return d, scaler, X, y, windowSize

from tensorflow.keras.layers import Conv1D, MaxPooling1D

def buildModel(input_shape):
    model = Sequential()
    
    # Conv1D layer
    model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())  # Add Batch Normalization
    model.add(MaxPooling1D(pool_size=2))  # Downsampling(reducing complexity)
    
    # Bidirectional LSTM 1st layer
    model.add(Bidirectional(LSTM(128, return_sequences=True)))  # More units and sequences
    model.add(Dropout(0.2))  # Dropout for regularization
    
    # Second LSTM layer with return_sequences=False
    model.add(Bidirectional(LSTM(128, return_sequences=False)))  # More units and return sequences
    model.add(Dropout(0.2))
    
    # Dense layers for more feature learning
    model.add(Dense(256, activation='relu'))  # Increased units in dense layer
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu'))  # Another dense layer
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(1))  # Prediction of anomaly score
    
    # Compile the model with adam optimizer and MSE loss
    model.compile(optimizer='adam', loss='mse')
    return model


def detectAnom(model, X, y_true):
    y_pred = model.predict(X, verbose=0)
    mse = np.mean(np.square(y_pred - y_true), axis=1)

    # Normalize the MSE anomaly scores to range [0, 1]
    scoreScaler = MinMaxScaler()
    normalizedScores = scoreScaler.fit_transform(mse.reshape(-1, 1)).flatten()

    return normalizedScores

def main(csv_path, output_path):
    d, scaler, X, y, windowSize = loadAndPreprocess(csv_path)

    model = buildModel((X.shape[1], X.shape[2]))
    earlyStop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=30, batch_size=32, callbacks=[earlyStop], verbose=1)

    normalizedScores = detectAnom(model, X, y)

    # Add anomaly scores back to the original csvfile
    anomalyScores = [np.nan] * windowSize + list(normalizedScores)

    d['anomaly_score'] = anomalyScores

    d.to_csv(output_path, index=False)
    print(f"[INFO] Normalized anomaly scores saved to: {output_path}")



if __name__ == '__main__':
    csv_path = r'C:\Users\osobh\GUC\Bachelor Thesis\NAB\data\realAWSCloudwatch\rds_cpu_utilization_e47b3b.csv'
    output_path = r'C:\Users\osobh\GUC\Bachelor Thesis\lstm results\lstm_final_results.csv'
    main(csv_path, output_path)
