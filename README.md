# neuralNetModeler
Welcome!

This is a Python program that models a neural network.

Currently the network will load a set of data from a CSV into a helper Matrix object which is then loaded into the Network object.

Set your layers, count of neurons per layer, the type of activation function (relu, tanh, and sigmoid supported), learning rate, how many epochs to run, and this will 
process the training set for as many times as epochs is loaded.

With debug on your CLI will update progress on training.

There is still plenty to do with this project. The intent is for myself to learn the internals of how the networks move data, back propagate, and learn from the data.

This was built as a part of my MIT Professional Education: From Data to Decisions with Machine Learning and to support the impact project DRONET.

A few outstanding TODOs:
1. TODO: save state of training neurons to a file for later use. This will allow us to save the trained model and use it for predictions without retraining.
2. TODO: implement a method to load a trained model from a file and use it for predictions.
3. TODO: complete matrix analysis to suggest a number of layers and neurons per layer based on the input data.
4. TODO: shuffle the input data for each epoch to ensure that the training is not biased by the order of the data.
5. TODO: Implement softmax
6. TODO: Implement testing

Example call from CLI:

> py main.py input=drone_sar_synthetic_data.csv activation=relu training=true trainingcolumn=8 usesuggested=true epochs=500

Example usage can be seen in main.py.

or :
```python

from utility.matrix import Matrix

from network.network import Network

inputFile = File(path="data/training",name=parameters.getParameter('input'))



neuralNetwork = Network(True) #initial setup with debug turned on

inputFile.setColumnsToIngore([0,1])  #this will tell it to ignore certain columns in the data.

data = inputFile.readCSVIntoMatrix(True)

dataMatrix = Matrix(data) #creates a Matrix object

dataMatrix.normalize() #normalizes the data so that all datapoints are between 0 and 1.

neuralNetwork.setMatrix(dataMatrix) #load our Matrix object of data to process

neuralNetwork.featuresToIgnore([0,1]) #if there are features in the data to ignore

neuralNetwork.setTrainingColumn(8) #tells the Network we are training and which column in the Matrix holds our training label.

neuralNetwork.setEpochs(500) #how many times will the network process the entire dataset.

neuralNetwork.setLearningRate(0.01)

neuralNetwork.setBias(0.2) #initial bias

neuralNetwork.setOutputLayer(1,'sigmoid') #how wide is the output layer and what is its activation function

neuralNetwork.initialize(layerCount,neuronsPerLayerCount,'relu') #this will set off the creation of the layers and each layer its neurons with the given parameters.
    
neuralNetwork.process() #starts processing.
    
#TODO neuralNetwork.getPredicionRate() #retrieve the final prediction rate
   
#TODO neuralNetwork.save() #save the network state.
```

