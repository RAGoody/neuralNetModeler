from neuron.neuron import neuron

class layer:
    debug = False
    layer = 0   #which layer this is in depth
    neuronsPerLayerCount = 0 #how many neurons we have per layer
    neurons = [] #each position is a neuron.
    weights = [] #parallel to self.neurons, stores the weight of the corresponding neuron
    bias = []
    activation = 'relu'
    input = []
    inputSet = False
    incomingConnections = 0
    outgoingConnections = 0

    def __init__(self,layer,neuronsPerLayerCount,activation,debug=False):
        """
            Sets out attributes <boolean>debug and 
            <integer>layer depth this instance is with 0 being Input and the highest number Ouput, all between Hidden.
        """
        self.debug = debug
        self.layer = layer
        self.neuronsPerLayerCount = neuronsPerLayerCount
        self.incomingConnections = neuronsPerLayerCount #placeholder for later modifications to adjust layer width.
        self._createLayer(neuronsPerLayerCount,activation)

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
        outputs = []
        for i in range(self.neuronsPerLayerCount):
            self.neurons[i].setInput(self.activation)
            outputs.append(self.neurons[i].calculate())

        print(outputs)

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

            self.neurons.append(neuron())
            if (self.debug == True):
                print(self.neurons[i])

            self.neurons[i].setConnections(self.incomingConnections)
            self.neurons[i].setActivation(activation)
            self.neurons[i].setBias(0)