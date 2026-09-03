from neuron.neuron import Neuron
import uuid

class Layer:
    debug = False
    layer = 0   #which layer this is in depth
    neuronsPerLayerCount = 0 #how many neurons we have per layer
    activation = 'relu'
    inputSet = False
    learningRate = 0.01
    staticBias = 0.1
    useStaticBias = False
    incomingConnections = 0
    outgoingConnections = 0

    def __new__(cls, *args, **kwargs):
        # Force the creation of a completely new object instance
        instance = super().__new__(cls)
        return instance 

    def __init__(self,layer,neuronsPerLayerCount,incomingConnections,activation,learningRate,debug=False):
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
        self.learningRate = learningRate
        self.neuronsPerLayerCount = neuronsPerLayerCount
        self.incomingConnections = incomingConnections
        self.activation = activation

    def create(self):
        self._createLayer(self.neuronsPerLayerCount,self.activation,self.learningRate)

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

    def getOutput(self):
        return self.output

    def process(self):
        output = []
        #current iterative N Neurons
        for i in range(self.neuronsPerLayerCount):
            self.neurons[i].setInput(self.input)
            output.append(self.neurons[i].calculate())

        self.output = output
        #if (self.debug == True):
            #print(f"Layer: {self.layer} has generated output: {self.output}")

        return self.output

    def isInputSet(self):
        return self.inputSet

    def getNeuron(self,number):
        return self.neurons[number]

    def learn(self,trainingIndicator,predictedProbability):
        for i in range(self.neuronsPerLayerCount):
            self.neurons[i].learn(trainingIndicator,predictedProbability[i])

    def calculateErrors(self,comparisonLayer):
        for thisNeuronIndex in range(self.neuronsPerLayerCount):
            thisDelta = 0
            thisWeight = 0
            for thisComparisonNeuron in range(comparisonLayer.getneuronsPerLayerCount()):
                thisDelta += comparisonLayer.getNeuron(thisComparisonNeuron).getDelta()
                thisWeight += comparisonLayer.getNeuron(thisComparisonNeuron).getSpecificWeight(thisNeuronIndex)

            self.neurons[thisNeuronIndex].calculateHiddenDelta(thisDelta,thisWeight)

        for thisNeuronIndex in range(self.neuronsPerLayerCount):
            self.neurons[thisNeuronIndex].adjustWeights()

    def getWeightsAtIndex(self,index):
        weights = []
        for thisNeuronIndex in range(self.neuronsPerLayerCount):
            weights.append(self.neurons[thisNeuronIndex].getSpecificWeight(index))

        return weights

    def getNeurons(self):
        return self.neurons

    def getneuronsPerLayerCount(self):
        return self.neuronsPerLayerCount

    def setDebug(self,debug):
        self.debug = debug

    def setActivation(self):
        for i in range(self.neuronsPerLayerCount):
            self.neurons[i].setActivation(self.activation)

    def setBias(self):
        for i in range(self.neuronsPerLayerCount):
            self.neurons[i].setBias(self.bias[i])

    def generateBias(self):
        for i in range(self.inputLen):
            self.bias[i] = 0

    def setStaticBias(self,bias):
        self.staticBias = bias
        self.useStaticBias = True

    def _createLayer(self,neuronsPerLayerCount,activation,learningRate):
        for i in range(neuronsPerLayerCount):
            self.neurons.append(Neuron())
            self.neurons[i].setConnections(self.incomingConnections)
            self.neurons[i].setActivation(activation)
            self.neurons[i].setLearningRate(learningRate)
            if (self.useStaticBias == True):
                self.neurons[i].setBias(self.staticBias)
            else:
                self.neurons[i].setBias(self.bias[i])

            if (self.debug == True):
                print(f"........Created neuron {i}. {self.neurons[i].getId()}. Activation: {self.neurons[i].getActivation()}. Learning Rate: {self.neurons[i].getLearningRate()}. Bias: {self.neurons[i].getBias()}. Connections: {self.neurons[i].getConnections()}")