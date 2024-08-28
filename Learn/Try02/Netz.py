import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

class ChurnModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = tf.keras.models.load_model("Learn\Try02\netz.keras")

    def load_data(self):
        self.df = pd.read_csv(self.file_path)

    def preprocess_data(self):
        self.X = pd.get_dummies(self.df.drop(['Churn', 'Customer ID'], axis=1))
        self.y = self.df['Churn'].map({'Yes': 1, 'No': 0})

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def define_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(self.X_train.shape[1],)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):
        checkpoint = ModelCheckpoint('Netz.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        self.model.fit(self.X_train, self.y_train, epochs=200, batch_size=32, verbose=2)

    def evaluate_model(self):
        y_hat = self.model.predict(self.X_test)
        y_hat = (y_hat > 0.5).astype(int)
        accuracy = accuracy_score(self.y_test, y_hat)
        print(f'Test accuracy: {accuracy:.3f}')

    def save_model(self):
        self.model.save('Netz.keras')

    def load_saved_model(self):
        del self.model
        self.model = load_model('Netz.keras')
    
    def configure_model(self):
        print("Konfigurieren Sie das neuronale Netz:")
        num_layers = int(input("Wie viele Layer möchten Sie haben? "))
        num_neurons = []
        for i in range(num_layers):
            num_neurons.append(int(input(f"Wie viele Neuronen möchten Sie im Layer {i+1} haben? ")))
        activation_functions = []
        for i in range(num_layers):
            activation_functions.append(input(f"Welche Aktivierungsfunktion möchten Sie im Layer {i+1} haben? (relu, sigmoid, tanh) "))
        num_epochs = int(input("Wie viele Epochen möchten Sie trainieren? "))
        batch_size = int(input("Wie groß soll die Batch-Größe sein? "))
        test_size = float(input("Wie groß soll der Testdatensatz sein? (0.0 - 1.0) "))

        self.model = Sequential()
        for i in range(num_layers):
            if i == 0:
                self.model.add(Dense(num_neurons[i], activation=activation_functions[i], input_shape=(self.X_train.shape[1],)))
            else:
                self.model.add(Dense(num_neurons[i], activation=activation_functions[i]))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.train_model(num_epochs, batch_size, test_size)

    def train_model(self, num_epochs, batch_size, test_size):
        checkpoint = ModelCheckpoint('Netz.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        self.model.fit(self.X_train, self.y_train, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_split=test_size)
