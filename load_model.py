import data_reader
import training

import mlflow.pyfunc


def main():
    # Wczytanie modelu z MLFlow
    model_uri = "models:/RepairCostModel/1"
    model = mlflow.pyfunc.load_model(model_uri)

    # Funkcja do przewidywania kosztów
    def predict_cost(features):
        return model.predict(features)

    # Przykład użycia modelu w systemie
    new_repair = [[0.5, 0.3, 0.2]]  # Przykładowe dane nowego zgłoszenia
    predicted_cost = predict_cost(new_repair)
    print(f'Przewidywany koszt naprawy: {predicted_cost[0]}')


if __name__ == '__main__':
    print("xdd")
    # main()
