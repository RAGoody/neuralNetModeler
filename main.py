"""
Orchestration program for modeling a neural net.
Expects:
    a file within the data/ directory named <input>.csv     data to process in CSV format with a header
    a file within the data/ directory named <input>.json    json file of weights to assign to columns in <input>.csv
CLI parameters:
    <string>input - what file contains input. used in both training and predicting modes.
    <string>activation - which activation function math to use. Valid inputs: 'relu','sigmoid','tanh','softmax'. Softmax non operable.
    <integer>layers - how many layers to use.
    <integer>neuronsperlayer - nuerons per layer to use.
    <boolean>usesuggested - are we using suggested layercounts and neuroncounts or use the given CLI parameters for each?
    <boolean>training   - are we training the model?
    <boolean>useconfig  - non operable

"""

from utility.file import file
from utility.cli import cli
from utility.matrix import matrix
from network.network import network
import sys
from readchar import readkey, key

parameters = cli(parameters=['input', 'output', 'activation', 'layers','neuronsperlayer','usesuggested','training','useconfig'])
print(parameters.getParameters())

#read our config - set aside for now in lieu of intaking parameters from the CLI

#init our empty network with debug turned on.
neuralNetwork = network(True)
neuralNetwork.featuresToIgnore([0,1])

if (parameters.getParameter('training') == True):
    inputFile = file(path="data/training",name=parameters.getParameter('input'))
    print("Training mode detected.")
else:
    inputFile = file(path="data/input", name=parameters.getParameter('input'))
    print("Predictive mode detected.")

#read our data
print(f"Reading input file: {inputFile.fullPath}")

#TODO : update the colums to ignore to either CLI params for user inputs.
if (parameters.getParameter('training') == True):
    inputFile.setColumnsToIngore([0,1,10])  #columns 0 and 1 are lat/long and 10 is if there's a target present.
else:
    inputFile.setColumnsToIngore([0,1])  #columns 0 and 1 are lat/long and 10 is if there's a target present.
data = inputFile.readCSVIntoMatrix(True)
dataMatrix = matrix(data)
dataMatrix.normalize()
neuralNetwork.setMatrix(dataMatrix)

#determine if we're using forced depth and neuron count, or if we're going to use what the object suggests.
if (parameters.getParameter('usesuggested') == True):
    neuralNetwork.analyzeInputMatrix()
    layerCount = neuralNetwork.getSuggestedLayers()
    neuronsPerLayerCount = neuralNetwork.getSuggestedNeurons()
    print(f"Suggested layers: {layerCount}")
    print(f"Suggested Neurons per layer: {neuronsPerLayerCount}")
    print("Continue? Press 'enter' or 'escape' to quit.")
    key = readkey()
    if (key == "\x1b"):
        print("exiting. Goodbye.")
        print("")
        sys.exit()
    else:
        print("...Continuing.")
else:
    layerCount = int(parameters.getParameter('layers'))
    neuronsPerLayerCount = int(parameters.getParameter('nueronsperlayer'))

activation = parameters.getParameter('activation')
activation.lower()

neuralNetwork.setOutputLayer(1,'sigmoid')
#neuralNetwork.setIterationBreak(1)
neuralNetwork.initialize(layerCount,neuronsPerLayerCount,activation)
neuralNetwork.process()
