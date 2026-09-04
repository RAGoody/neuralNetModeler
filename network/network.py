from matplotlib.pylab import rint
from layer.layer import Layer
import math

class Network:
    """
        This class handles the orchestration of a neural network. It manages layers, neurons, and the flow of data through the network.
        Current state: will trainin on a data set, output its progress as it learns for the observer to see if it is learning or not. 
        TODO: save state of training neurons to a file for later use. This will allow us to save the trained model and use it for predictions without retraining.
        TODO: implement a method to load a trained model from a file and use it for predictions.
        TODO: complete matrix analysis to suggest a number of layers and neurons per layer based on the input data.
            Currently suggests 3 layers and a number of neurons equal to the number of features in the input data.
        Example Usage:
            neuralNetwork = Network(True)
            neuralNetwork.featuresToIgnore([0,1])
            neuralNetwork.setTrainingColumn(8)
            neuralNetwork.setEpochs(500)
            neuralNEtwork.setLearningRate(0.01)
            neuralNetwork.setOutputLayer(1,'sigmoid')
            neuralNetwork.initialize(layerCount,neuronsPerLayerCount,'relu')
            neuralNetwork.process()
    """
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
    epochs = 1
    accuracyThreshold = 0.9
    learningRate = 0.01 #just a default value.

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
        activation.lower()

        if (self.isOutputSet() == False):
            if (self.debug == True):
                print("No output layer parameters set. Please use Network.setOutputLayer(width<int>,activation<str>) before calling Network.initialize()")
            return False

        if (self.setActivation(activation) == True):
            self._createNet(layerCount,neuronsPerLayerCount,self.activation,self.learningRate)
        else:
            raise TypeError (f"Neuron::isValidActivation: '{activation}' is not a valid activation. Valid inputs: 'relu','sigmoid','tanh','softmax'.")

    def setTrainingColumn(self,columnNumber):
        """
            In training mode we need to know which column in our matrix is indicating the expected output for the training data. This is where we set that column number.
            Also we initailize our boolean training mode to control processing behaviors and a list to hold the training indicators for each row of data.
        """
        self.trainingColumn = columnNumber
        self.trainingMode = True
        self.trainingValues = []

    def setIterationBreak(self,count):
        # Is there a point we want to stop streaming data to review debug output?
        self.iterationBreak = count

    def setEpochs(self,epochs):
        self.epochs = epochs

    def setLearningRate(self,learningRate):
        self.learningRate = learningRate

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

    def setLearningRate(self,learningRate):
        self.learningRate = learningRate

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
        self.suggestedLayers = 3
        self.suggestedNeurons = self.matrix.getWidth()
        if (self.trainingMode == True):
            self.suggestedNeurons -= 1  #make sure to reduce suggested neuron count by 1 since we expect a column to be our training reference.

        return True

    def process(self,iteration=0):
        """Send the dataset into the network for processing"""
        if (self.initialized == False):
            return False

        accuracyOverTimeList = []
        self.processStarted = True
        matrixLength = self.matrix.getLength()
        for i in range(self.epochs): #an epoch is how many times we will run our training dataset.
            accuracyList = []
            lossList = []
            #TODO: test for a threshold of correct predictions & break once acheived.
            if (self.debug == True):
                print(f"Network:process(): Epoch {i+1} of {self.epochs}", end=" ")

            for row in range(matrixLength):
                #if (self.debug == True):
                    #print(".", end=" ")
                
                #pass data to layer 0.
                input = self.matrix.getRowAsList(row)
                #print(f"    Network:process() after getting row: {row} input: {input}")

                #handle training column
                if (self.trainingMode == True):
                    trainingIndicator = input[self.trainingColumn] #save our training indicator for the loss function.
                    self.trainingValues.append(trainingIndicator)  #this becomes a list parallel to the length of self.matrix
                    del input[self.trainingColumn] #pop our training column out of the data.

                #print(f"    2nd Network:process() after training column handling: input: {input}")

                self.layers[0].setInput(input)
                priorOutput = self.layers[0].process()
                #print(f"    3rd Network:process(): after 0th layer process input: {priorOutput}")

                #move data forward through the network.
                for layerIndex in range(self.layerCount):
                    if layerIndex > 0:
                        temp = priorOutput
                        #print(f"    Network:process(): before layer {layerIndex} processes")
                        #print(f"        input{priorOutput}")
                        self.layers[layerIndex].setInput(priorOutput)
                        priorOutput = self.layers[layerIndex].process()
                        #print(f"    after layer {layerIndex} processes")
                        #print(f"        output{priorOutput}")

                    #if (i == self.layerCount-1):
                        #print(f"layer:process():sigmoidlayer input for this layer:{temp}")

                #calculate loss with our training indicator when in training mode.
                if (self.trainingMode == True):
                    loss = self.processLoss(priorOutput,trainingIndicator)
                    lossList.append(loss[0])
                    if (trainingIndicator == 1):
                        accuracy = trainingIndicator - priorOutput[0]
                        accuracy = 1 - abs(accuracy)
                        accuracyList.append(accuracy)

                    #if (self.debug == True):
                        #print(f"{self.layers[self.layerCount-1].getOutput()} vs. {priorOutput}")
                        #print(f"        Loss for this layer: {loss} with training indicator of: {trainingIndicator} and last layer prediction: {priorOutput}")

                    if (row > 0):
                        self.layers[self.layerCount-1].learn(trainingIndicator,priorOutput)
                        self.calculateHiddenErrors()

                if (row == self.iterationBreak):
                    if (self.debug == True):
                        print(f"    Network:process():Breaking! at row {row}")
                    break

            if (self.debug == True):
                print("")
                
            if (len(accuracyList) > 0):
                averageAccuracy = sum(accuracyList) / len(accuracyList)
                averageLoss = sum(lossList) / len(lossList)
                if (self.debug == True):
                    print(f"    Network:process(): Average Accuracy for Epoch {i+1}: {averageAccuracy}")
                    print(f"    Network:process(): Average Loss for Epoch {i+1}: {averageLoss}")

                accuracyOverTimeList.append(averageAccuracy)
                if (averageAccuracy >= self.accuracyThreshold and self.trainingMode == True):
                    if (self.debug == True):
                        print(f"Network:process(): Average Accuracy of {averageAccuracy} has met or exceeded threshold of {self.accuracyThreshold}. Breaking training loop.")
                    break

        self.processed = True
        if (self.debug == True):
            print(f"Network:process(): Average Accuracy over all epochs: {sum(accuracyOverTimeList) / len(accuracyOverTimeList)}")

        return True

    def calculateHiddenErrors(self):
        """
            Since Network is the only object aware of multiple layers and the output, it will need to orchestrate each layer calculating 
            its neurons delta from the subsequenct layer.
            I COULD just straight-up iterate through each neuron in two sets but I'd break my encapsulation.
        """
        #print("......Network:calculateHiddenErrors()")
        for layerIndex in reversed(range(self.layerCount)):
            nextLayerIndex = layerIndex - 1
            if (nextLayerIndex > 0):
                #lets grab our bottom-most layer in the iteration.
                closestToBottomLayer = self.layers[layerIndex]
                #now grab the next layer up in the network.
                closestToTopLayer = self.layers[nextLayerIndex]
                #each neuron in the next up layer needs to calculate how off down-stream calculations were because of it.
                closestToTopLayer.calculateErrors(closestToBottomLayer)

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
                    thisLoss = -(indicator * math.log(probability) + (1.0 - indicator) * math.log(1.0 - probability))
                case 'MSE':
                    #Mean Squared Error. For use with ReLU
                    errorDifference = predictedProbability - indicator
                    squaredError = errorDifference ** 2
                    thisLoss = 0.5 * squaredError
            loss.append(thisLoss)

        #print(f"thisoutput: {output} thisLoss: {loss}")
        return loss

    def _createNet(self,layerCount,neuronsPerLayerCount,activation,learningRate):
        """
            Orchestrates creating the whole net. Uses parameters to build the incoming and hidden layers.
            Uses separate attributes self.outputWidth & self.outputActivation to build the output layer.
            Parameters:
                layerCount: <int> how many layers to build.
                neuronsPerLayerCount <int> how many neurons are in this layer
                activation <str> which activation function are we using in this layer?
                learningRate <float> the amount of adjustment we want this layer to change its weights by.
        """
        for i in range(layerCount):
            self._initializeLayer(i,neuronsPerLayerCount,neuronsPerLayerCount,activation,learningRate)

        #now set the output layer. This is always additive to the set # of layers
        self._initializeLayer(layerCount,self.outputWidth,neuronsPerLayerCount,self.outputActivation,learningRate)
        self.layerCount = len(self.layers)
        self.initialized = True

    def _initializeLayer(self,number,neuronsPerLayerCount,incomingConnections,activation,learningRate):
        """
            initializes a single layer & updates our layer list attribute with the new layer.
            Parameters:
                number: <int> what layer this is.
                neuronsPerLayerCount <int> how many neurons are in this layer
                incomingConnections <int> how many connections are coming into this layer or how many features are we processing
                activation <str> which activation function are we using in this layer?
                learningRate <float> the amount of adjustment we want this layer to change its weights by.
        """
        if (self.debug == True):
            print(f"....Creating layer {number} with {neuronsPerLayerCount} neurons.")

        self.layers.append(Layer(number,neuronsPerLayerCount,incomingConnections,activation,learningRate,self.debug))
        self.layers[number].setStaticBias(self.bias)
        self.layers[number].create()
        if (self.debug == True):
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