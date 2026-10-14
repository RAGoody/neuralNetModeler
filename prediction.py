"""
Author: R. A. Goodwin
Date: August 2026 - September 2026
Orchestration program for TRAINING our model neural net.
This code is fully human-generated and is not AI-generated.
It is designed to read in a CSV file of training data, configure a neural network based on user parameters, 
and either train the model or make predictions based on the input data.
Expects:
    TODO: a file within the data/ directory named <input>.json    json file of weights to assign to columns in <input>.csv
    training data set within data/training/ directory named <input>.csv     data to process in CSV format with a header
CLI parameters:
    input (str) - what file contains input. used in both training and predicting modes.
    model (str) - what file contains the model state. 
    name (str) - what name to use for the model state file.
    columnstoignore (str) - which columns to ignore in the CSV file. Comma separated list of integers.
    
Example usage:
py prediction.py input=drone_sar_synthetic_data.csv model=network_state_2026-09-21_16-35-32.json columnstoignore=0,1
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

parameters = CLI(parameters=['input', 'model', 'name', 'columnstoignore'])
print(parameters.getParameters())

parameters.setRequiredParameters(['input', 'model', 'columnstoignore'])

stateFile = File(path="data/input", name=parameters.getParameter('model'))
print(f"Predictive mode detected. Loading latest state from {stateFile.fullPath} (expected in JSON format).")
stateFile.read()
networkState = json.loads(stateFile.getContents())
print(f"....... Now loading state file from {stateFile.fullPath}.")

#read our data
inputFile = File(path="data/input",name=parameters.getParameter('input'))
print(f"Reading input file: {inputFile.fullPath}")
inputFile.setColumnsToIngore(parameters.getParameter('columnstoignore').split(','))
data = inputFile.readCSVIntoMatrix(True)
dataMatrix = Matrix(data)
dataMatrix.normalize()

#Create our Network. At this point it is completely empty.
neuralNetwork = Network(True)

#Load the data to run our predictions against.
neuralNetwork.setMatrix(dataMatrix)

#Load the state of the network from the JSON file. This will set up the network with the correct weights and biases.
neuralNetwork.loadState(networkState)