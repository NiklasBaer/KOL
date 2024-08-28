from Netz import ChurnModel
import pandas as pd
import tensorflow as tf
import os

def load_model(model_path):
    if os.path.exists(r"Learn\Try02\netz.keras"):
        model = ChurnModel(r"Learn\Try02\netz.keras")
        return model
    else:
        return None

def make_prediction(model):
    print("Machen Sie eine Vorhersage:")
    customer_id = input("Geben Sie die Kunden-ID ein: ")
    data = pd.DataFrame({'Customer ID': [customer_id]})
    data = pd.get_dummies(data)
    prediction = model.model.predict(data)
    print(f"Die Vorhersage für den Kunden {customer_id} ist: {prediction[0]}")

def train_model(model_path):
    print("Trainieren Sie das Netzwerk:")
    model = ChurnModel(model_path)
    model.configure_model()
    model.train_model()
    return model

def main():
    model_path = os.path.join(os.getcwd(), "Learn", "Try02", "tfmodel.keras")
    model = load_model(model_path)

    if model is not None:
        while True:
            print("Wählen Sie eine Option:")
            print("1. Machen Sie eine Vorhersage")
            print("2. Trainieren Sie das Netzwerk")
            print("3. Beenden")
            choice = input("Geben Sie Ihre Wahl ein: ")
            if choice == "1":
                make_prediction(model)
            elif choice == "2":
                model = train_model(model_path)
            elif choice == "3":
                break
            else:
                print("Ungültige Wahl. Bitte wählen Sie erneut.")
    else:
        print("Kein Modell gespeichert. Bitte trainieren Sie das Netzwerk.")
        model = train_model(model_path)
        print("Modell gespeichert. Bitte starten Sie das Programm neu.")

if __name__ == "__main__":
    main()