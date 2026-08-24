import numpy as np
import math
import random

class neuron:
    weights = []
    bias = 0
    output = False
    input = 0
    incomingConnections = 0
    outgoingConnections = 0
    activation = 'relu'

    def __init__(self):
        #whatever we need to initialize. Purposefully leaving neuron clueless to the outside world at init.
        a = 1

    def setInput(self,input):
        #TODO: validation on input and understanding the type of input.
        self.input = input

    def getOutput(self):
        return self.output

    def setActivation(self,activation):
        if (self._isValidActivation == True):
            self.activation = activation
            self._generateWeights()
            return True
        else:
            return False

    def setConnections(self,connections):
        self.incomingConnections = connections
        self.outgoingConnections = connections

    def getConnections(self):
        return self.connections

    def setBias(self,bias):
        self.bias = bias

    def getBias(self):
        return self.bias

    def getActivation(self):
        return self.activation

    def _isValidActivation(self,activation):
        """
           Is this a valid activation function type?
        """
        match activation:
            case 'relu' | 'sigmouid' | 'tanh' | 'softmax':
                return True
            case _ : 
                raise TypeError (f"Neuron::isValidActivation: '{activation}' is not a valid activation. Valid inputs: 'relu','sigmoid','tanh','softmax'.")

    def _generateWeights(self):
        """
           Based upon the activation type, will generate an initial weight for processing.
        """
        match self.activation:
            case 'relu':
                # He Initilization for ReLU
                self.weights = [random.gauss(0,1) * math.sqrt(2.0 / self.connections) for i in range(self.incomingConnections) ]
                return True
            case 'sigmoid' | 'tanh':
                #Xavier uniform 
                limit = np.sqrt(6.0 / self.incomingConnections + self.outgoingConnections)
                self.weights = [np.random.uniform(-limit, limit, size=(self.incomingConnections, self.outgoingConnections)) for i in range.self.incomingConnections]
                return True
            case 'softmax':
                #placeholder
                b = 4
                return True
            case _:
                return False

    def calculate(self):
        """
            Processes input based upon selected activation function.
            Returns False if invalid activation function selected.
        """

        #TODO: NEED TYPE HANDLING WHEN READING THE CSV

        match self.activation :
            case 'relu' :
                if (self.input > 0) :
                    self.output = self.input + self.bias
                else:
                    self.output = 0
            case 'sigmoid':
                self.output = np.where(self.input >= 0, 
                    1 / (1 + np.exp(-self.input)), 
                    np.exp(self.input) / (1 + np.exp(self.input))) + self.bias
            case 'tanh' :
                self.output = np.tanh(np.array([self.input])) + self.bias
            case 'softmax' :
                self.output = self.input

        return self.output