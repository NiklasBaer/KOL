
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
                                                                    #* import von Modulen und Datensetzt
                                                              
X, Y = spiral_data(100,3)                                           #* Test daten erstellen dur eine Spiralfunktion 


class Layer_Dense:                                                  #? Klasse Layer erstellen
    
    def __init__(self, n_inputs, n_neurons):
        
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  #* festlegen wie viele Inputs und Neuroen wir haben

        self.biases = np.zeros((1, n_neurons))                      #* biases alles = 0
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases    #* brechnet den Output jedes einzelnen Neurones
        
class Activatiopn_ReLU:                                             #? Klasse Activtion erstellen hier legen wir fest wann jedes Einzellen Neuron activ ist oder nicht mit ein ReLU
    
    def forward(self, inputs):
        
        self.output = np.maximum(0, inputs)                         #* Alles werste die unter 0 liegen werden auf eine 0 gesetz
   
   
   
     
layer1 = Layer_Dense(2,5)                                           #* Erstellung des Objektes Layer1 mit 2 inputs und 5 Neuronen
activation1 = Activatiopn_ReLU()                                    #* Ertsellung des Objektes activation1

layer1.forward(X)                                                   #* Layer1 mit der Funktion forward ausführen und mit den Argumenten X (input wertde)
activation1.forward(layer1.output)                                  #* Nun alles werte unter 0 auf 0 setzten

print(activation1.output)                                           # * outputs ausgeben auf der Console