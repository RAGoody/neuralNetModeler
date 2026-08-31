from layer.layer import layer

class network:
    debug = False
    layers = []
    layerCount = 0
    bias = 0
    neuronsPerLayerCount = 0
    matrix = []
    matrixLoaded = False
    initialized = False
    processed = False
    processStarted = False
    activation = False
    depthProcessed = 0
    suggestedNeurons = 0
    suggestedLayers = 0
    processingLayer = 0 #which layer is currently processing.
    trainingColumn = 0

    def __init__(self,debug=False):
        #initial setup
        self.debug = debug

    def initialize(self,layerCount,neuronsPerLayerCount,activation):
        self.layerCount = layerCount
        self.neuronsPerLayerCount = neuronsPerLayerCount
        if (self.setActivation(activation) == True):
            self._createNet(layerCount,neuronsPerLayerCount,self.activation)
        else:
            raise TypeError (f"Neuron::isValidActivation: '{activation}' is not a valid activation. Valid inputs: 'relu','sigmoid','tanh','softmax'.")

    def setTrainingColumn(self,columnNumber):
        self.trainingColumn = columnNumber

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

    def setBias(self,bias):
        self.bias = bias

    def getBias(self):
        return self.bias

    def featuresToIgnore(self,ignoreList):
        """
            If there are column features to ignore in our matrix, this is where we tell the network.
        """
        if (type(ignoreList) == list):
            self.ignoreList = ignoreList
        else:
            self.ignoreList = [ignoreList]

        return True

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

    def getSuggestedLayers(self):
        return self.suggestedLayers

    def getSuggestedNeurons(self):
        return self.suggestedNeurons

    def iterateThroughEachNeuron(self,action='debug'):
        for x in range(self.layerCount):
            for y in range(self.neuronsPerLayerCount):
                thisLayer = self.getLayer(x)
                match action:
                    case 'debug':
                        print(f"Iterating through layer {x}: {thisLayer}")
                        thisNeuron = thisLayer.getNeuron(y)
                        print(f"...displaying neuron {y} of layer {x}: {thisNeuron}")
                    case 'setActivation':
                        thisLayer.getNeuron(y).setActivation(self.activation)
                    case 'setActAndBias':
                        thisLayer.getNeuron(y).setActivation(self.activation)
                        thisLayer.getNeuron(y).setBias(self.bias)


    def setMatrix(self,matrix):
        if (type(matrix) == 'utility.matrix.matrix'):
            print("Network:analyzeInputData(): Parameter is not the right type of <matrix> for the network to operate.")
            return False
        else:
            self.matrix = matrix
            self.matrixLoaded = True
            return True

    def isMatrixLoaded(self):
        return self.matrixLoaded

    def getMatrixRow(self,row):
        if (self.isMatrixLoaded == True):
            return self.matrix[row]
            
    def analyzeInputMatrix(self):
        """
            Reviews the included data, suggest a number of neurons and layers to process the data set.
            Parameters:
                data<matrix>
            Returns:
                <boolean>
        """
        #for thisLine in data:
            #print(thisLine)
        
        self.suggestedLayers = 5
        self.suggestedNeurons = self.matrix.getWidth()

        return True

    def process(self,iteration=0):
        """Send the dataset into the network for processing"""
        if (self.initialized == False):
            return False
        
        self.processStarted = True

        matrixLength = self.matrix.getLength()
        for row in range(matrixLength):
            #pass data to layer 0.
            self.layers[0].setInput(self.matrix.getRowAsList(row))
            priorOutput = self.layers[0].process()

            #move data forward through the network.
            for i in range(self.layerCount):
                if i > 0:
                    self.layers[i].setInput(priorOutput)
                    priorOutput = self.layers[i].process()

            #TODO : add in output handling logic here. First version will inlude a probability calculation.
            # 1 layer of a neuron configured for sigmoid that will take the output from the last layer of the network
            # then calculate a probability for that row of matrix data.

        #TODO: add in Loss calculation after processed comparing result versus the training data column.

        #TODO: adjust bias and weights based on correctness.

        self.processed = True
        return True

    def hasProcessed(self):
        return self.processed
    
    def _createNet(self,layerCount,neuronsPerLayerCount,activation):
        for i in range(layerCount):
            self.initializeLayer(i,neuronsPerLayerCount,activation)

        self.initialized = True

    def _iterateThroughNLayers(self,maxDepth=1):
        depthCounter = 0
        for x in range(self.layerCount):
            for y in range(self.neuronsPerLayerCount):
                thisLayer = self.getLayer(x)
                print(f"Iterating through layer {x}: {thisLayer}")
            depthCounter += 1
            if (depthCounter == maxDepth):
                break