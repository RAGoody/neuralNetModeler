from layer.layer import layer

class network:
    debug = False
    layers = []
    layerCount = 0
    neuronsPerLayerCount = 0
    data = []
    initialized = False
    processed = False
    processStarted = False
    activation = False

    def __init__(self,layerCount,neuronsPerLayerCount,activation,debug=False):
        #initial setup
        self.debug = debug
        self.layerCount = layerCount
        self.neuronsPerLayerCount = neuronsPerLayerCount
        if (self.setActivation(activation) == True):
            self._createNet(layerCount,neuronsPerLayerCount,self.activation)
        else:
            raise TypeError (f"Neuron::isValidActivation: '{activation}' is not a valid activation. Valid inputs: 'relu','sigmoid','tanh','softmax'.")

    def setDebug(self,debug):
        self.debug = debug

    def setActivation(self,activation):
        if (self._isValidActivation(activation) == True):
            self.activation = activation
            return True
        else:
            return False

    def getActivation(self):
        return self.activation

    def _isValidActivation(self,activation):
        match activation:
            case 'relu' | 'sigmoid' | 'tanh' | 'softmax':
                print("valid activation")
                return True
            case _ :
                return False

    def initializeLayer(self,number,neuronsPerLayerCount,activation):
        if (self.debug == True):
            print(f"....Creating layer {number} with {neuronsPerLayerCount} neurons.")

        self.layers.append(layer(number,neuronsPerLayerCount,activation,self.debug))

    def getLayer(self,number):
        if (number > len(self.layers)):
            return False

        return self.layers[number]

    def getLayers(self):
        return self.layers

    def showLayer(self,number):
        if (len(self.layers) > number):
            return False
        print(self.layers[number])

    def iterateThroughEachNeuron(self,action='debug'):
        for x in range(self.layerCount):
            for y in range(self.neuronsPerLayerCount):
                match action:
                    case 'debug':
                        thisLayer = self.getLayer(x)
                        print(f"Iterating through layer {x}: {thisLayer}")
                        thisNeuron = thisLayer.getNeuron(y)
                        print(f"...displaying neuron {y} of layer {x}: {thisNeuron}")

    def process(self,data):
        """Send the dataset into the network for processing"""
        if (self.initialized == False):
            return False
        
        self.processStarted = True
        return self.processedStarted

    def hasProcessed(self):
        return self.processed
    
    def _createNet(self,layerCount,neuronsPerLayerCount,activation):
        for i in range(layerCount):
            self.initializeLayer(i,neuronsPerLayerCount,activation)

        self.initialized = True