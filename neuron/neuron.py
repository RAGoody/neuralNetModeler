import numpy as np
import math
import random
import uuid

class Neuron:
    bias = 0
    output = False
    input = 0
    incomingConnections = 0
    activation = 'dud'
    id = ''
    delta = 0
    learningRate = 0.001

    def __new__(cls, *args, **kwargs):
        # Force the creation of a completely new object instance
        instance = super().__new__(cls)
        return instance 

    def __init__(self):
        #whatever we need to initialize. Purposefully leaving neuron clueless to the outside world at init.
        self.weights = []
        self.lastInput = []
        self.id = uuid.uuid4()

    def setInput(self,input):
        #TODO: validation on input and understanding the type of input.
        self.input = input

    def getOutput(self):
        return self.output

    def getId(self):
        return self.id

    def setLearningRate(self,learningRate):
        self.learningRate = learningRate

    def getLearningRate(self):
        return self.learningRate

    def setActivation(self,activation):
        if (self._isValidActivation(activation) == True):
            self.activation = activation
            self._generateWeights()
            return True
        else:
            raise ValueError (f"Invalid activation: '{activation}'. Use 'relu','sigmoid', or 'tanh'")

    def setConnections(self,connections):
        self.incomingConnections = connections

    def getConnections(self):
        return self.incomingConnections

    def setBias(self,bias):
        self.bias = bias

    def getBias(self):
        return self.bias

    def getActivation(self):
        return self.activation

    def calculate(self):
        """
            Processes input based upon selected activation function.
            Returns False if invalid activation function selected.
        """
        total = 0
        self.lastInput = self.input
        for i in range(self.incomingConnections):
            total += float(self.input[i]) * self.weights[i]
      
        match self.activation :
            case 'relu' :
                total += self.bias
                if (total > 0) :
                    self.output = total
                else:
                    self.output = 0.01 * total
            case 'sigmoid':
                total += self.bias
                if (total > 500):
                    self.output = 1
                elif (total < -500):
                    self.output = 0
                else:                
                    self.output = float(1 / (1 + np.exp(-total)))
            case 'tanh' :
                self.output = np.tanh(np.array([self.input])) + self.bias
            case 'softmax' :
                #TODO : not operable at the moment.
                self.output = total

        return self.output

    def learn(self,trainingIndicator,predictedProbability):
        self.delta = predictedProbability - trainingIndicator
        #print(f"            Neuron:learn() : delta: {self.delta}")
        self.adjustWeights()
        return self.delta

    def getDelta(self):
        return self.delta

    def calculateHiddenDelta(self,delta,weight):
        if (self.output > 0):
            composite = delta * weight
            self.delta = composite * 1.0
        else:
            self.delta = 0 * 0.01

    def adjustWeights(self):
        """
            Adjusts the weights for each incoming feature based upon the learningRate * what the last input was for this neuron.
        """
        self.bias -= self.learningRate * self.delta

        """
        print ("...Neuron.adjustWeights() ==================================================")
        print (f"{self.id}")
        print (f"weights before: {self.weights}")
        print (f"last inputs before: {self.lastInput}")
        print (f"delta: {self.delta}")
        print (f"learning: {self.learningRate}")
        print (f"output: {self.output}")
        print (f"activation: {self.activation}")
        """
        for i in range(self.incomingConnections):
            self.weights[i] -= self.learningRate * self.delta * self.lastInput[i]

        """
        print (f"weights after: {self.weights}")
        print ("==================================================")
        """

    def getSpecificWeight(self,index):
        return self.weights[index]
 
    def _isValidActivation(self,activation):
        """
           Is this a valid activation function type?
        """
        match activation:
            case 'relu' | 'sigmoid' | 'tanh' | 'softmax':
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
                self.weights = [random.gauss(0,1) * math.sqrt(2.0 / self.incomingConnections) for i in range(self.incomingConnections) ]
                return True
            case 'sigmoid' | 'tanh':
                #Xavier uniform 
                limit = np.sqrt(6.0 / (self.incomingConnections + 1))
                self.weights = np.random.uniform(-limit, limit, size=self.incomingConnections).tolist()
            case 'softmax':
                #placeholder
                b = 4
                return True
            case _:
                return False

    def _cleanValue(self,value):
        #TODO: assess this method is needed in neuron.

        useTemp = False
        match value.lower(): #Are ya a boolean disguised as a silly string!?
            case 'true': 
                return True #now go away or I shall taunt you-ah ah second time-ah
            case 'false':
                return False #now go away or I shall taunt you-ah ah second time-ah
        
        try: #Are you an integer??
            temp = int(value) 
            useTemp = True #Yes? a hopeful true, to be overridden in the except on a bad attempt.
        except: 
            useTemp = False #No! Not really an integer.

        if (useTemp == True):
            return temp #off the bridge you go!
            
        try: #Arrrrre you a floater??
            temp = float(value)
            useTemp = True #Yes? a hopeful true, to be overridden in the except on a bad attempt.
        except:
            useTemp = False #No! Not really a floater.

        if (useTemp == True):
            return temp #off the bridge you go!

        #Failing the type conversions to boolean, int, & float, then just return the parameter. It's probably a string.
        return value