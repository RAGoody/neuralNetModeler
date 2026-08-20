"""
Orchestration program for modeling a neural net.
Expects:
    a file within the data/ directory named <input>.csv     data to process in CSV format with a header
    a file within the data/ directory named <input>.json    json file of weights to assign to columns in <input>.csv
CLI parameters:
    input

"""

from utility.file import file
from utility.cli import cli
from network.network import network
import sys

parameters = cli(parameters=['input', 'output', 'activation', 'layers','nueronsperlayer'])
print(parameters.getParameters())

layerCount = int(parameters.getParameter('layers'))
neuronsPerLayerCount = int(parameters.getParameter('nueronsperlayer'))
activation = parameters.getParameter('activation')
activation.lower()

neuralNetwork = network(layerCount,neuronsPerLayerCount,activation,True)

neuralNetwork.iterateThroughEachNeuron('debug')

for i in range(layerCount):
    neuralNetwork.showLayer(i)