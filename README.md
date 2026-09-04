# neuralNetModeler
Welcome!

This is a Python program that models a neural network.

Currently the network will load a set of data from a CSV into a helper Matrix object which is then loaded into the Network object.

Set your layers, count of neurons per layer, the type of activation function (relu, tanh, and sigmoid supported), learning rate, how many epochs to run, and this will 
process the training set for as many times as epochs is loaded.

With debug on your CLI will update progress on training.

Lots of cleanup and tweaking needed, but is currently trainable.


Example call from CLI:

> py main.py input=drone_sar_synthetic_data.csv activation=relu training=true trainingcolumn=8 usesuggested=true epochs=500

Example usage can be seen in main.py.

or :
```python
neuralNetwork = Network(True) #initial setup with debug turned on

neuralNetwork.setMatrix(dataMatrix) #load our Matrix object of data to process

neuralNetwork.featuresToIgnore([0,1]) #if there are features in the data to ignore

neuralNetwork.setTrainingColumn(8) #tells the Network we are training and which column in the Matrix holds our training label.

neuralNetwork.setEpochs(500)

neuralNetwork.setLearningRate(0.01)

neuralNetwork.setBias(0.2) #initial bias

neuralNetwork.setOutputLayer(1,'sigmoid') #how wide is the output layer and what is its activation function

neuralNetwork.initialize(layerCount,neuronsPerLayerCount,'relu') #this will set off the creation of the layers and each layer its neurons with the given parameters.
    
neuralNetwork.process() #starts processing.
    
#TODO neuralNetwork.getPredicionRate() #retrieve the final prediction rate
   
#TODO neuralNetwork.save() #save the network state.
```

