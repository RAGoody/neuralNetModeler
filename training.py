"""
Author: R. A. Goodwin
Date: August 2026 - September 2026
Orchestration program for training our model a neural net.
This code is fully human-generated and is not AI-generated.
It is designed to read in a CSV file of training data, configure a neural network based on user parameters, 
and either train the model or make predictions based on the input data.
Expects:
    TODO: a file within the data/ directory named <input>.json    json file of weights to assign to columns in <input>.csv
    training data set within data/training/ directory named <input>.csv     data to process in CSV format with a header
CLI parameters:
    input (str) - what file contains input. used in both training and predicting modes.
    activation (str) - which activation function math to use. Valid inputs: 'relu','sigmoid','tanh','softmax'. Softmax non operable.
    outputactivation (str) - which activation function math to use for the output layer. Valid inputs: 'relu','sigmoid','tanh','softmax'. Softmax non operable.
    outputwidth (int) - how many neurons to use in the output layer.
    usesuggested (bool) - are we using suggested layercounts and neuroncounts or use the given CLI parameters for each?
    layers (int) - how many layers to use.
    neuronsperlayer (int) - neurons per layer to use.
    bias (float) - what bias to use for the neurons in the network.
    learningrate (float) - what learning rate to use for training.
    trainingcolumn (int) - which column contains the training labels?
    epochs (int) - how many epochs to train for.

Example usage:
py training.py input=drone_sar_synthetic_data.csv activation=relu outputactivation=sigmoid outputwidth=1 bias=0.2 learningrate=0.001 trainingcolumn=8 usesuggested=true epochs=200 columnstoignore=0,1

"""
#Package Imports
from datetime import datetime
import json
from random import random
import sys
from readchar import readkey, key

#Custom Imports
from utility.file import File
from utility.cli import CLI
from utility.matrix import Matrix
from network.network import Network

parameters = CLI(parameters=['input','activation','outputactivation','outputwidth','layers','neuronsperlayer','usesuggested','trainingcolumn','learningrate','epochs','name','columnstoignore'])
print(parameters.getParameters())

parameters.setRequiredParameters(['input', 'activation', 'outputactivation', 'outputwidth', 'trainingcolumn','learningrate','epochs','name','bias'])

activation = parameters.getParameter('activation')
outputActivation = parameters.getParameter('outputactivation')
outputWidth = parameters.getParameter('outputwidth')
trainingColumn = parameters.getParameter('trainingcolumn')
learningRate = parameters.getParameter('learningrate')
epochs = parameters.getParameter('epochs')
name = parameters.getParameter('name')
bias = parameters.getParameter('bias')

#Set our input file path and name using our CLI parameters. This is where the training data lives.
inputFile = File(path="data/training",name=parameters.getParameter('input'))

#If there are columns to ignore in the CSV, then we tell our file object to ignore those columns during reading the file.
#This prevents polluting the Matrix with unnecessary data.
if (parameters.getParameter('columnstoignore') != None):
    columnsToIgnore = parameters.getParameter('columnstoignore').split(",")
    columnsToIgnore = [int(x) for x in columnsToIgnore]  #convert to integers
    print(f"Columns to ignore: {columnsToIgnore}")
    inputFile.setColumnsToIngore(columnsToIgnore)

#Read the CSV file into a Matrix object. The Matrix class will handle normalization and other matrix operations.
print(f"Reading input file: {inputFile.fullPath}")
data = inputFile.readCSVIntoMatrix(True)
dataMatrix = Matrix(data)
dataMatrix.normalize()

#Create our Network. At this point it is completely empty.
neuralNetwork = Network(True)

#Load our normalized data into the neural network. The network will use this data for training.
neuralNetwork.setMatrix(dataMatrix)
neuralNetwork.setTrainingColumn(trainingColumn) #this needs to be set before we analyze the input matrix, as it will ignore that column when analyzing the data.

if (parameters.getParameter('usesuggested') == True):
    neuralNetwork.analyzeInputMatrix()
    layerCount = neuralNetwork.getSuggestedLayers()
    neuronsPerLayerCount = neuralNetwork.getSuggestedNeurons()
elif (parameters.getParameter('usesuggested') == False):
    if (parameters.getParameter('layers') == None or parameters.getParameter('neuronsperlayer') == None):
        print("Error: You must provide both layers and neuronsperlayer parameters if not using suggested values.")
        sys.exit(1)
    layerCount = parameters.getParameter('layers')
    neuronsPerLayerCount = parameters.getParameter('neuronsperlayer')

#Set our Network Parameters & initialize.
print(f"Initializing Network with parameters: ")
print(f"activation={activation}")
print(f"outputActivation={outputActivation}")
print(f"outputWidth={outputWidth}")
print(f"trainingColumn={trainingColumn}")
print(f"learningRate={learningRate}")
print(f"epochs={epochs}")
print(f"bias={bias}")
print(f"layerCount={layerCount}")
print(f"neuronsPerLayerCount={neuronsPerLayerCount}")
neuralNetwork.setOutputLayer(outputWidth,outputActivation)
neuralNetwork.setBias(bias)
neuralNetwork.setLearningRate(learningRate)
neuralNetwork.setEpochs(epochs)
neuralNetwork.initialize(layerCount,neuronsPerLayerCount,activation) #At this point our network is fully initialized and ready to train.

print(f"Network Initialized: {neuralNetwork}")
print("Continue? Press 'enter' or 'escape' to quit.")
key = readkey()
if (key == "\x1b"):
    print("exiting. Goodbye.")
    print("")
    sys.exit()
else:
    print("...Continuing.")

#Now we process our training data, each row of the matrix will be processed through the network, and the weights and biases will be adjusted within the Network based upon
#the set activation function, learning rate, and other parameters.
neuralNetwork.process()

print(f"Processing completed")
print("Save state? Press any key to save or 'escape' to quit.")
key = readkey()
if (key == "\x1b"):
    print("exiting. Goodbye.")
    print("")
    sys.exit()

networkState = neuralNetwork.getState()
print(f"Network State: {networkState}")
networkStateJSON = json.dumps(networkState)

todaysDate = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
stateFile = File(path="data/output",name=f"network_state_{todaysDate}.json")
stateFile.write(networkStateJSON,True)