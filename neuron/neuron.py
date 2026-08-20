import numpy as np

class neuron:
    csvColumnTitle = ''
    csvColumnCount = 0
    bias = False
    output = False
    input = 0
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
            return True
        else:
            return False

    def setBias(self,bias):
        self.bias = bias

    def getBias(self):
        return self.bias

    def getActivation(self):
        return self.activation

    def _isValidActivation(self,activation):
        match activation:
            case 'relu' | 'sigmouid' | 'tanh' | 'softmax':
                return True
            case _ : 
                raise TypeError (f"Neuron::isValidActivation: '{activation}' is not a valid activation. Valid inputs: 'relu','sigmoid','tanh','softmax'.")
                return False

    def _calculate(self):
        """
            Processes input based upon selected activation function.
            Returns False if invalid activation function selected.
        """
        match self.activation :
            case 'relu' :
                if (self.input > 0) :
                    self.output = self.input
                else:
                    self.output = 0
            case 'sigmoid':
                self.output = np.where(self.input >= 0, 
                    1 / (1 + np.exp(-self.input)), 
                    np.exp(self.input) / (1 + np.exp(self.input))) 
            case 'tanh' :
                self.output = np.tanh(np.array([self.input]))
            case 'softmax' :
                self.output = self.input

        return self.output