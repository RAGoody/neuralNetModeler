from neuron.neuron import Neuron
import uuid

class Layer:
    """
    Author: R. A. Goodwin
    Date: August 2026 - September 2026
    This code is fully human-generated and is not AI-generated.
    A class to represent a layer in a neural network. Each layer contains multiple neurons and manages the flow of data through them.
    Attributes:
        id (str): A unique identifier for the layer.
        neurons (list): A list of Neuron objects contained in the layer.
        weights (list): A list of weights for the connections to the next layer.
        bias (list): A list of bias values for the neurons in the layer.
        input (list): The input values to the layer.
        output (list): The output values from the layer after processing.
        debug (bool): A flag to enable or disable debug output.
        layer (int): The depth of the layer in the network (0 for input, highest for output).
        neuronsPerLayerCount (int): The number of neurons in the layer.
        activation (str): The activation function type for the neurons in the layer ('relu', 'sigmoid', 'tanh', 'softmax').
        inputSet (bool): A flag indicating whether the input has been set for the layer.
        learningRate (float): The learning rate for weight adjustments in the layer.
        staticBias (float): A static bias value to be applied to all neurons in the layer if useStaticBias is True.
        useStaticBias (bool): A flag indicating whether to use a static bias for all neurons in the layer.
        incomingConnections (int): The number of incoming connections to the layer.
        outgoingConnections (int): The number of outgoing connections from the layer. #NOT IN USE & NEEDS TO BE REMOVED.
        activation (str): The activation function type for the neurons in the layer ('relu', 'sigmoid', 'tanh', 'softmax').
    """
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
        Sets out attributes
            debug (bool)
            layer (int) depth this instance is with 0 being Input and the highest number Ouput, all in between Hidden.
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

    def setWeights(self,weights):
        for thisNeuronIndex in range(self.neuronsPerLayerCount):
            self.neurons[thisNeuronIndex].setWeights(weights[thisNeuronIndex])

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

    def getState(self):
        """
            Returns the current state of the layer. This will allow us to save the trained model and use it for predictions without retraining.
            State is defined as the weights and biases of each neuron in the layer. This will be a list of lists. 
            Each neuron is a list of weights and a bias.
        """
        state = []
        for neuron in self.neurons:
            neuron_state = {
                'weights': neuron.getWeights(),
                'bias': neuron.getBias()
            }
            state.append(neuron_state)
        return state

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