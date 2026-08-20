from neuron.neuron import neuron

class layer:
    debug = False
    layer = 0   #which layer this is in depth
    neuronsPerLayerCount = 0 #how many neurons we have per layer
    neurons = [] #each position is a neuron.
    weights = [] #parallel to self.neurons, stores the weight of the corresponding neuron
    activation = 'relu'

    def __init__(self,layer,neuronsPerLayerCount,activation,debug=False):
        """
            Sets out attributes <boolean>debug and 
            <integer>layer depth this instance is with 0 being Input and the highest number Ouput, all between Hidden.
        """
        self.debug = debug
        self.layer = layer
        self.neuronsPerLayerCount = neuronsPerLayerCount
        self._createLayer(neuronsPerLayerCount,activation)

    def getLayerNumber(self):
        return self.layer

    def setLayerNumber(self,layer):
        self.layer = layer

    def getNeuron(self,number):
        return self.neurons[number]

    def getNeurons(self):
        return self.neurons

    def setDebug(self,debug):
        self.debug = debug

    def _createLayer(self,neuronsPerLayerCount,activation):
        for i in range(neuronsPerLayerCount):
            if (self.debug == True):
                print(f"........Creating neuron {i}.")
            self.neurons.append(neuron())
            if (self.debug == True):
                print(self.neurons[i])
            self.neurons[i].setActivation(activation)