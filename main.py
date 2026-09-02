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

from utility.file import File
from utility.cli import CLI
from utility.matrix import Matrix
from network.network import Network
import sys
from readchar import readkey, key

parameters = CLI(parameters=['input', 'output', 'activation', 'layers', 'neuronsperlayer', 'usesuggested', 'training', 'trainingcolumn', 'useconfig'])
print(parameters.getParameters())

#read our config - set aside for now in lieu of intaking parameters from the CLI
#TODO: implement a Config class that reads stores a JSON configuration with User Parameters in it.

#init our empty network with debug turned on.
neuralNetwork = Network(True)
neuralNetwork.featuresToIgnore([0,1])

if (parameters.getParameter('training') == True):
    inputFile = File(path="data/training",name=parameters.getParameter('input'))
    print("Training mode detected.")
else:
    inputFile = File(path="data/input", name=parameters.getParameter('input'))
    print("Predictive mode detected.")

#read our data
print(f"Reading input file: {inputFile.fullPath}")

inputFile.setColumnsToIngore([0,1])  #columns 0 and 1 are lat/long and 10 is if there's a target present.
data = inputFile.readCSVIntoMatrix(True)
dataMatrix = Matrix(data)
dataMatrix.normalize()
neuralNetwork.setMatrix(dataMatrix)
if (parameters.getParameter('training') == True):
    neuralNetwork.setTrainingColumn(parameters.getParameter('trainingcolumn'))  #remember to account for any ignored columns.

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
