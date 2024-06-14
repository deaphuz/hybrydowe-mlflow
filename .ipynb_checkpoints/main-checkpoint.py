import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import mlflow.tensorflow

mlflow.set_experiment('Repair Cost Prediction2')
mlflow.set_tracking_uri("sqlite:///mlflow.db")
data = pd.read_csv('failures_data.csv')

print(data.head())
print(data.info())
print(data.describe())

# Czyszczenie danych
data.dropna(inplace=True)

data['FAILURE_TYPE'] = data['FAILURE_TYPE'].astype('category').cat.codes
data['STATUS'] = data['STATUS'].astype('category').cat.codes

data['DATE'] = pd.to_datetime(data['DATE'], format='%Y-%m-%d').astype('int64') / 10**9
data['POTENTIAL_DATA'] = pd.to_datetime(data['POTENTIAL_DATA'], format='%Y-%m-%d').astype('int64') / 10**9

# Ekstrakcja cech
features = data[['FAILURE_TYPE', 'DATE', 'POTENTIAL_PRICE', 'POTENTIAL_DATA', 'STATUS']]
labels = data['POTENTIAL_PRICE']


def augment_data(df, num_duplicates=100):
    augmented_data = []
    for i in range(num_duplicates):
        for index, row in df.iterrows():
            new_row = row.copy()

            new_row['POTENTIAL_PRICE'] += np.random.randint(-10000, 10000)
            if new_row['POTENTIAL_PRICE'] < 100:
                new_row['POTENTIAL_PRICE'] = 100
            new_row['DATE'] += np.random.randint(-1000000, 1000000)
            new_row['POTENTIAL_DATA'] += np.random.randint(-1000000, 1000000)
            augmented_data.append(new_row)
    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([df, augmented_df], ignore_index=True)

augmented_df = augment_data(data, num_duplicates=100)

# normalizacja
scaler = StandardScaler()
numerical_features = ['POTENTIAL_PRICE', 'DATE', 'POTENTIAL_DATA']
augmented_df[numerical_features] = scaler.fit_transform(augmented_df[numerical_features])
features_scaled = scaler.fit_transform(features)

# Display the augmented DataFrame
print(augmented_df)

def generate_additional_data(features, labels, n=100):
    new_features = []
    new_labels = []
    for _ in range(n):
        idx = np.random.randint(0, len(features))
        new_features.append(features[idx] + np.random.normal(0, 0.1, features.shape[1]))
        new_labels.append(labels[idx])
    return np.array(new_features), np.array(new_labels)

additional_features, additional_labels = generate_additional_data(features_scaled, labels)
features_final = np.vstack([features_scaled, additional_features])
labels_final = np.hstack([labels, additional_labels])


def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),  # Use Input to define the input shape
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Trenowanie i ewaluacja modelu
input_dim = features_final.shape[1]
model = build_model(input_dim)

# Logowanie modelu w MLFlow
with mlflow.start_run():
    mlflow.tensorflow.autolog()

    # Train the model
    history = model.fit(features_final, labels_final, validation_split=0.2, epochs=100,
                        callbacks=[EarlyStopping(patience=10)], verbose=1)

    # Save the model
    mlflow.keras.log_model(model, 'model')

model.summary()