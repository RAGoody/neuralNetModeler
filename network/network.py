from layer.layer import Layer
import math

class Network:
    debug = False
    layerCount = 0
    bias = 0
    neuronsPerLayerCount = 0
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
    trainingMode = False
    outputWidth = 0
    outputActivation = ''
    outputSet = False
    lossSet = False
    id = ''
    iterationBreak = -1

    def __init__(self,debug=False):
        #initial setup
        self.debug = debug
        #these lists are initialized here here due to Python's object reference handling. Setting a List attribute will often tell Python,
        #on subsequent initilizations, to create a pointer to the first object created rather than allocate new memory to a new object.
        self.layers = [] #a list of our layers
        self.matrix = [] #this ultimately becomes a Matrix object.

    def initialize(self,layerCount,neuronsPerLayerCount,activation):
        self.layerCount = layerCount
        self.neuronsPerLayerCount = neuronsPerLayerCount

        if (self.isOutputSet() == False):
            if (self.debug == True):
                print("No output layer parameters set. Please use Network.setOutputLayer(width<int>,activation<str>) before calling Network.initialize()")
            return False

        if (self.setActivation(activation) == True):
            self._createNet(layerCount,neuronsPerLayerCount,self.activation)
        else:
            raise TypeError (f"Neuron::isValidActivation: '{activation}' is not a valid activation. Valid inputs: 'relu','sigmoid','tanh','softmax'.")

    def setTrainingColumn(self,columnNumber):
        self.trainingColumn = columnNumber
        self.trainingMode = True
        self.trainingValues = []

    def setIterationBreak(self,count):
        # Is there a point we want to stop streaming data to review debug output?
        self.iterationBreak = count

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

    def setOutputLayer(self,width,activation):
        self.outputWidth = width
        self.outputActivation = activation
        self.outputSet = True
        match activation: #which loss fucntion we use is determined by what our output activation function is.
            case 'relu':
                self.setLossFunction('MSE')
            case 'sigmoid':
                self.setLossFunction('BCE')

        return True

    def isOutputSet(self):
        return self.outputSet

    def setLossFunction(self,type):
        match type :
            case 'BCE' | 'MSE':
                self.lossType = type
                self.lossSet = True
                return True
            case _:
                return False
            
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
        self.suggestedLayers = 5
        self.suggestedNeurons = self.matrix.getWidth()
        if (self.trainingMode == True):
            self.suggestedNeurons -= 1  #make sure to reduce suggested neuron count by 1 since we expect a column to be our training reference.

        return True

    def process(self,iteration=0):
        """Send the dataset into the network for processing"""
        if (self.initialized == False):
            return False
     
        self.processStarted = True

        matrixLength = self.matrix.getLength()
        for row in range(matrixLength):
            #pass data to layer 0.
            input = self.matrix.getRowAsList(row)

            #handle training column
            if (self.trainingMode == True):
                trainingIndicator = input[self.trainingColumn] #save our training indicator for the loss function.
                self.trainingValues.append(trainingIndicator)  #this becomes a list parallel to the length of self.matrix
                del input[self.trainingColumn] #pop our training column out of the data.

            self.layers[0].setInput(input)
            priorOutput = self.layers[0].process()

            #move data forward through the network.
            for i in range(self.layerCount):
                if i > 0:
                    self.layers[i].setInput(priorOutput)
                    priorOutput = self.layers[i].process()

            if (row == self.iterationBreak):
                if (self.debug == True):
                    print("Breaking!")
                break

            #calculate loss with our training indicator when in training mode.
            if (self.trainingMode == True):
                loss = self.processLoss(self.layers[self.layerCount-1].getOutput(),trainingIndicator)
                #TODO: adjust bias and weights based on correctness

        self.processed = True
        return True

    def hasProcessed(self):
        return self.processed

    def processLoss(self,output,indicator):
        loss = []
        outputLen = len(output)

        for i in range(outputLen):
            predictedProbability = output[i]
            match self.lossType:
                case 'BCE':
                    #Binary Cross-Entropy. For use with Sigmoid.
                    epsilon = 1e-15  #safety value to prevent crashes on .log
                    probability = max(epsilon, min(1.0 - epsilon, predictedProbability)) #ensure prediction value to not break math
                    thisLoss = (indicator * math.log(probability) + (1.0 - indicator) * math.log(1.0 - probability))
                case 'MSE':
                    #Mean Squared Error. For use with ReLU
                    errorDifference = indicator - predictedProbability
                    squaredError = errorDifference ** 2
                    thisLoss = 0.5 * squaredError
            loss.append(thisLoss)

        if (self.debug == True):
            print(f"....Loss for this layer: {loss} with training indicator of: {indicator}")

        return loss

    def _createNet(self,layerCount,neuronsPerLayerCount,activation):
        for i in range(layerCount):
            self._initializeLayer(i,neuronsPerLayerCount,activation)

        #now set the output layer. This is always additive to the set # of layers
        self._initializeLayer(layerCount,self.outputWidth,self.outputActivation)
        self.layerCount = len(self.layers)

        self.initialized = True

    def _initializeLayer(self,number,neuronsPerLayerCount,activation):
        if (self.debug == True):
            print(f"....Creating layer {number} with {neuronsPerLayerCount} neurons.")

        self.layers.append(Layer(number,neuronsPerLayerCount,activation,self.debug))
        print(self.layers[number].getId())

    def _iterateThroughNLayers(self,maxDepth=1):
        depthCounter = 0
        for x in range(self.layerCount):
            for y in range(self.neuronsPerLayerCount):
                thisLayer = self.getLayer(x)
                print(f"Iterating through layer {x}: {thisLayer}")
            depthCounter += 1
            if (depthCounter == maxDepth):
                break

    def _isValidActivation(self,activation):
        match activation:
            case 'relu' | 'sigmoid' | 'tanh' | 'softmax':
                print("valid activation")
                return True
            case _ :
                return False