from neuron.neuron import Neuron
import uuid


class Layer:
    debug = False
    layer = 0   #which layer this is in depth
    neuronsPerLayerCount = 0 #how many neurons we have per layer
    activation = 'relu'
    inputSet = False
    incomingConnections = 0
    outgoingConnections = 0

    def __new__(cls, *args, **kwargs):
        # Force the creation of a completely new object instance
        instance = super().__new__(cls)
        return instance 

    def __init__(self,layer,neuronsPerLayerCount,activation,debug=False):
        """
            Sets out attributes <boolean>debug and 
            <integer>layer depth this instance is with 0 being Input and the highest number Ouput, all between Hidden.
        """
        self.id = f'Layer-{layer}-{uuid.uuid4()}'
        self.neurons = [] #each position is a neuron.
        self.weights = []
        self.bias = []
        self.input = []
        self.output = []
        self.debug = debug
        self.layer = layer
        self.neuronsPerLayerCount = neuronsPerLayerCount
        self.incomingConnections = neuronsPerLayerCount #placeholder for later modifications to adjust layer width.
        self._createLayer(neuronsPerLayerCount,activation)

    def getId(self):
        return self.id
    
    def getLayerNumber(self):
        return self.layer

    def setLayerNumber(self,layer):
        self.layer = layer

    def setInput(self,input):
        if (type(input) != list):    
            return False

        self.input = input
        self.inputLen = len(self.input)
        self.inputSet = True
        return True

    def process(self):
        output = []
        #current iterative N Neurons
        for i in range(self.neuronsPerLayerCount):
            self.neurons[i].setInput(self.input)
            output.append(self.neurons[i].calculate())

        self.output = output
        if (self.debug == True):
            print(f"Layer: {self.layer} has generated output: {self.output}")

        return self.output

    def isInputSet(self):
        return self.inputSet

    def generateBias(self):
        for i in range(self.inputLen):
            self.bias[i] = 0

    def generateWeights(self):
        for i in range(self.inputLen):
            b = 1
    
    def getNeuron(self,number):
        return self.neurons[number]

    def getNeurons(self):
        return self.neurons

    def setDebug(self,debug):
        self.debug = debug

    def setActivation(self):
        for i in range(self.neuronsPerLayerCount):
            self.neurons[i].setActivation(self.activation)

    def setBias(self):
        for i in range(self.neuronsPerLayerCount):
            self.neurons[i].setBias(self.bias[i])

    def _createLayer(self,neuronsPerLayerCount,activation):
        for i in range(neuronsPerLayerCount):
            if (self.debug == True):
                print(f"........Creating neuron {i}.")

            self.neurons.append(Neuron())
            if (self.debug == True):
                print(self.neurons[i].getId())

            self.neurons[i].setConnections(self.incomingConnections)
            self.neurons[i].setActivation(activation)
            self.neurons[i].setBias(0)