import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from mlflow.metrics import precision_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt


import mlflow.tensorflow
import tensorflow.keras

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.autolog()

data = pd.read_csv('failures_data.csv')
# drop name column, it's not a feature
data = data.drop('NAME', axis=1)
data = data.drop('STATUS', axis=1)

experiment_description = (
    "This is the repair cost prediction experiment. "
)

experiment_tags = {
    "project_name": "repair-cost-prediction",
    "store_dept": "cost-prediction",
    "team": "studencciaki-pb",
    "project_quarter": "Q2-2024",
    "mlflow.note.content": experiment_description,
}

#cost_prediction_experiment = client.create_experiment(
#    name="Repair Cost Prediction", tags=experiment_tags
#)

cost_prediction_experiment = mlflow.set_experiment('Repair Cost Prediction')
mlflow.set_experiment('Repair Cost Prediction')
run_name = "cost_prediction_test"
artifact_path = "cptest_path"

print(data.head())
print(data.info())
print(data.describe())

# features - X
# labels - Y

#konwersja danych na liczby
data.dropna(inplace=True)

data['FAILURE_TYPE'] = data['FAILURE_TYPE'].astype('category').cat.codes
#data['STATUS'] = data['STATUS'].astype('category').cat.codes

data['DATE'] = pd.to_datetime(data['DATE'], format='%Y-%m-%d').astype('int64') / 10**9
data['POTENTIAL_DATA'] = pd.to_datetime(data['POTENTIAL_DATA'], format='%Y-%m-%d').astype('int64') / 10**9

features = data.drop('POTENTIAL_PRICE', axis=1)
target = data['POTENTIAL_PRICE']

#podzial na zbior treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def augment_data(X, y, num_new_samples=100):
    new_X = []
    new_y = []
    for _ in range(num_new_samples):
        idx = np.random.randint(0, len(X))
        new_sample = X[idx] + np.random.normal(0, 0.1, X.shape[1])
        new_X.append(new_sample)
        new_y.append(y.iloc[idx] + np.random.normal(0, 0.1))  # Użycie iloc do indeksowania
    return np.array(new_X), np.array(new_y)

new_X_train, new_y_train = augment_data(X_train_scaled, y_train,100)
X_train_augmented = np.vstack((X_train_scaled, new_X_train))
y_train_augmented = np.hstack((y_train, new_y_train))

print(X_train_augmented)
print("ok")
print(y_train_augmented)

def plot_and_log_metrics(history):
    # Tworzenie wykresów uczenia i straty
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Wykres funkcji straty
    axs[0].plot(history.history['loss'], label='loss')
    axs[0].plot(history.history['val_loss'], label='val_loss')
    axs[0].set_title('Funkcja straty')
    axs[0].set_xlabel('Epoka')
    axs[0].set_ylabel('Strata')
    axs[0].legend()

    # Wykres MAE
    axs[1].plot(history.history['mae'], label='mae')
    axs[1].plot(history.history['val_mae'], label='val_mae')
    axs[1].set_title('Mean Absolute Error')
    axs[1].set_xlabel('Epoka')
    axs[1].set_ylabel('MAE')
    axs[1].legend()

    plt.tight_layout()

    # Zapisywanie wykresów
    fig_path = "training_metrics.png"
    fig.savefig(fig_path)
    plt.close(fig)

    # Logowanie wykresów w MLFlow
    mlflow.log_artifact(fig_path)

def build_and_train_model(X_train, y_train, X_test, y_test):
    #deklaracja sekwencyjna
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    #kompilacja modelu
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy', 'precision'])
    #proces uczenia

#                       v data   v labels
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, validation_data= (X_test, y_test))

  #  test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)


    # Przetworzenie predykcji na kategorie do obliczenia accuracy, precision i f1
    threshold = y_test.median()
    y_pred_class = (y_pred > threshold).astype(int)
    y_test_class = (y_test > threshold).astype(int)

    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score()
    f1 = f1_score()

    return model, history, mse, mae, accuracy, precision, f1


# Ustawienie ścieżki eksperymentu MLFlow
mlflow.set_experiment("repair_cost_prediction")

for i in range(1):
    with mlflow.start_run(run_name=run_name) as run:
        model, history, mse, mae, accuracy, precision, f1 = build_and_train_model(
            X_train_augmented, y_train_augmented, X_test_scaled, y_test
        )


        mlflow.keras.log_model(model, "model")
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("accuracy", float(accuracy))


        plot_and_log_metrics(history)

        # Logowanie wykresów uczenia
       # for key in history.history.keys():
      #      for epoch, value in enumerate(history.history[key]):
       #         mlflow.log_metric(key, value, step=epoch)

        # Logowanie parametrów modelu
        mlflow.log_params({"optimizer": "adam", "loss": "mse", "epochs": 50, "batch_size": 32})


model.summary()