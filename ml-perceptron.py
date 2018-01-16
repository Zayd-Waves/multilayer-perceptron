"""
 -----------------------------------------------------------------------
|                                                                       |
|   Class:          NeuralNetwork                                       |
|   Description:    A feed-forward neural network that makes use of     |
|                   backpropagation.                                    |
|                                                                       |
|   Authors:        Zayd Bille                                          |
|   Date:           10/26/2017                                          |
|                                                                       |
 -----------------------------------------------------------------------
"""

from scipy.special import expit as sigmoid
from sklearn.datasets import fetch_mldata
from datetime import datetime
import numpy as np
import random

class NeuralNetwork:

    """
     ----------------------------------------------------------------
    |                                                                |
    |                      ---------------                           |
    |                     | Class Members |                          |
    |                      ---------------                           |
    |                                                                |
    |          int numOut     - Number of neurons in the output      |
    |                           layer.                               |
    |                                                                |
    |          int numIn      - Number of neurons in the input       |
    |                           layer (the features, essentially).   |
    |                                                                |
    |        int numLayers    - Number of hidden layers. The         |
    |                           network supports a maximum of        |
    |                           three layers.                        |
    |                                                                |
    |    int numHiddenNeurons - Number of neurons in                 |
    |                           each hidden layer.                   |
    |                                                                |
    |    matrix weightMatrix1 - Matrix that holds the weights to     |
    |                           be applied from the input layer      |
    |                           to the hidden layer.                 |
    |                                                                |
    |    matrix weightMatrix2 - Matrix that holds the weights to     |
    |                           be applied from the hidden layer     |
    |                           to the output layer.                 |
    |                                                                |
    |    matrix weightMatrix3 - Matrix that holds the weights to     |
    |                           be applied from a hidden layer       |
    |                           to another hidden layer. Only        |
    |                           exists if numLayers > 1.             |
    |                                                                |
    |    matrix weightMatrix4 - Matrix that holds the weights to     |
    |                           be applied from a hidden layer       |
    |                           to another hidden layer. Only        |
    |                           exists if numLayers > 2              |
    |                                                                |
    |         int epochs      - A single epoch is one pass through   |
    |                           the whole training dataset.          |
    |                                                                |
    |        int foldSize     - Size of subsets for the k-fold       |
    |                           step.                                |
    |                                                                |
    |       int trainSize     - Amount of data designated for        |
    |                           training (60,000 for MNIST).         |
    |                                                                |
    |        int testSize     - Amount of data designated for        |
    |                           testing (10,000 for MNIST).          |
    |                                                                |
    |      int learningRate   - Learning rate.                       |
    |                                                                |
    |       int lambdaValue   - Lambda value.                        |
    |                                                                |
    |                                                                |
     ----------------------------------------------------------------
    """

    """
     ----------------------------------------------------------------
    |    function: __init__()                                        |
    |    --------------------                                        |
    |                                                                |
    |    Initialization function that creates an instance of         |
    |    the neural network.                                         |
     ----------------------------------------------------------------
    """
    def __init__(self, 
                 numOut = 10,
                 numIn = 784,
                 numLayers = 1,
                 numHiddenNeurons = 30,
                 epochs = 350,
                 foldSize = 1,
                 trainSize = 60000,
                 testSize = 10000,
                 learningRate = 0.001,
                 lambdaValue = 0.01):
        self.numOut = numOut
        self.numIn = 784
        if (numLayers > 3):
            self.numLayers = 3
        else:
            self.numLayers = numLayers
        self.numHiddenNeurons = numHiddenNeurons
        self.weightMatrix1 = None
        self.weightMatrix2 = None
        self.weightMatrix3 = None
        self.weightMatrix4 = None
        self.epochs = epochs
        self.foldSize = foldSize
        self.trainSize = trainSize
        self.testSize = testSize
        self.learningRate = learningRate
        self.lambdaValue = lambdaValue
    
    """
     ----------------------------------------------------------------
    |    function: chooseWeights()                                   |
    |    -------------------------                                   |
    |                                                                |
    |    Creates matrices filled with random weights for neurons     |
    |    in each layer.                                              |
     ----------------------------------------------------------------
    """
    def chooseWeights(self):
    
        weightSet1 = np.random.uniform(-1.0, 1.0, self.numHiddenNeurons * (self.numIn + 1))
        weightSet1 = weightSet1.reshape(self.numHiddenNeurons, self.numIn + 1)
               
        if (self.numLayers == 1):
            
            weightSet2 = np.random.uniform(-1.0, 1.0, self.numOut * (self.numHiddenNeurons))
            weightSet2 = weightSet2.reshape(self.numOut, self.numHiddenNeurons)
            
            return (weightSet1, weightSet2)
            
        elif (self.numLayers == 2):
            
            weightSet2 = np.random.uniform(-1.0, 1.0, self.numHiddenNeurons * (self.numHiddenNeurons))
            weightSet2 = weightSet2.reshape(self.numHiddenNeurons, (self.numHiddenNeurons))
            
            weightSet3 = np.random.uniform(-1.0, 1.0, self.numOut * (self.numHiddenNeurons))
            weightSet3 = weightSet3.reshape(self.numOut, self.numHiddenNeurons)
            
            return (weightSet1, weightSet2, weightSet3)
        
        elif (self.numLayers == 3):
            
            weightSet2 = np.random.uniform(-1.0, 1.0, self.numHiddenNeurons * (self.numHiddenNeurons))
            weightSet2 = weightSet2.reshape(self.numHiddenNeurons, (self.numHiddenNeurons))
            
            weightSet3 = np.random.uniform(-1.0, 1.0, self.numHiddenNeurons * (self.numHiddenNeurons))
            weightSet3 = weightSet3.reshape(self.numHiddenNeurons, (self.numHiddenNeurons))
            
            weightSet4 = np.random.uniform(-1.0, 1.0, self.numOut * (self.numHiddenNeurons))
            weightSet4 = weightSet4.reshape(self.numOut, self.numHiddenNeurons)
            
            return (weightSet1, weightSet2, weightSet3, weightSet4) 
        
    """
     ----------------------------------------------------------------
    |    function: sigmoid()                                         |
    |    -------------------                                         |
    |                                                                |
    |    Sigmoid function. Activation algorithm between layers.      |
     ----------------------------------------------------------------
    """
    def sigmoid(self, x):
        z = sigmoid(x)
        return z
        
    """
     ----------------------------------------------------------------
    |    function: feedForward()                                     |
    |    -----------------------                                     |
    |                                                                |
    |    Commpletes a single run of the feed-forward calculations.   |
     ----------------------------------------------------------------
    """
    def feedForward(self, data, w1, w2):
        
        # Calculate input layer with bias/threshold added.
        inputLayer = np.ones((data.shape[0], data.shape[1] + 1))
        inputLayer[:, 1:] = data            
   
        # Matrix of the sums of the input layer's output multiplied
        # by each weight. Input of the hidden layer. 
        # (input layer -> hidden layer)
        sum1 = w1.dot(inputLayer.T)
        
        # Activation of the hidden layer.
        hiddenLayer = self.sigmoid(sum1)
        
        
        # Matrix of the sums of the hidden layer's output multiplied
        # by each weight. Input of the output layer. 
        # (hidden layer -> output layer)
        sum2 = w2.dot(hiddenLayer)
        
        # Activation of the ouput layer.
        outputLayer = self.sigmoid(sum2)
        
        return inputLayer, sum1, hiddenLayer, sum2, outputLayer
        
    """
     ----------------------------------------------------------------
    |    function: feedForwardML()                                   |
    |    -----------------------                                     |
    |                                                                |
    |    Commpletes a single run of the feed-forward calculations.   |
    |    Used with muliple hidden layers.                            |
     ----------------------------------------------------------------
    """
    def feedForwardML(self, data, w1, w2, w3, w4):
        
        if (self.numLayers == 2):
            # Calculate input layer with bias/threshold added.
            inputLayer = np.ones((data.shape[0], data.shape[1] + 1))
            inputLayer[:, 1:] = data            
       
            # Matrix of the sums of the input layer's output multiplied
            # by each weight. Input of the hidden layer. 
            # (input layer -> hidden layer)
            sum1 = w1.dot(inputLayer.T)
            
            # Activation of the first hidden layer.
            hiddenLayer1 = self.sigmoid(sum1)
            
            # Matrix of the sums of the hidden layer's output multiplied
            # by each weight. Input of the output layer. 
            # (hidden layer -> hidden layer 2)
            sum2 = w2.dot(hiddenLayer1)
            
            # Activation of the second hidden layer.
            hiddenLayer2 = self.sigmoid(sum2)
            
            # Matrix of the sums of the second hidden layer's output multiplied
            # by each weight. Input of the output layer. 
            # (hidden layer 2 -> output layer)
            sum3 = w3.dot(hiddenLayer2)
            
            # Activation of the ouput layer.
            outputLayer = self.sigmoid(sum3)
            
            return inputLayer, sum1, hiddenLayer1, sum2, hiddenLayer2, sum3, outputLayer
            
        if (self.numLayers == 3):
            # Calculate input layer with bias/threshold added.
            inputLayer = np.ones((data.shape[0], data.shape[1] + 1))
            inputLayer[:, 1:] = data            
       
            # Matrix of the sums of the input layer's output multiplied
            # by each weight. Input of the hidden layer. 
            # (input layer -> hidden layer)
            sum1 = w1.dot(inputLayer.T)
            
            # Activation of the first hidden layer.
            hiddenLayer1 = self.sigmoid(sum1)
            
            
            # Matrix of the sums of the hidden layer's output multiplied
            # by each weight. Input of the output layer. 
            # (hidden layer -> hidden layer 2)
            sum2 = w2.dot(hiddenLayer1)
            
            # Activation of the second hidden layer.
            hiddenLayer2 = self.sigmoid(sum2)
            
            # Matrix of the sums of the second hidden layer's output multiplied
            # by each weight. Input of the output layer. 
            # (hidden layer 2 -> output layer)
            sum3 = w3.dot(hiddenLayer2)
            
            # Activation of the third hidden layer.
            hiddenLayer3 = self.sigmoid(sum3)
            
            # Matrix of the sums of the third hidden layer's output multiplied
            # by each weight. Input of the output layer. 
            # (hidden layer 2 -> output layer)
            sum4 = w4.dot(hiddenLayer3)
            
            # Activation of the ouput layer.
            outputLayer = self.sigmoid(sum4)
            
            return inputLayer, sum1, hiddenLayer1, sum2, hiddenLayer2, sum3, hiddenLayer3, sum4, outputLayer
            
    """
     ----------------------------------------------------------------
    |    function: loadTrainData()                                   |
    |    -------------------------                                   |
    |                                                                |
    |    Loads matrices containing the mnist training data images    |
    |    and labels. Includes 60,000 images and labels.              |
     ----------------------------------------------------------------
    """
    def loadTrainData(self):
        mnist = fetch_mldata('MNIST original')
        
        sets = np.arange(len(mnist.data))
        
        trainIndex = np.arange(0, self.trainSize)

        images, labels = mnist.data[trainIndex], mnist.target[trainIndex]
        
        return images, labels

    """
     ----------------------------------------------------------------
    |    function: loadTestData()                                    |
    |    ------------------------                                    |
    |                                                                |
    |    Loads matrices containing the mnist testing data images     |
    |    and labels. Includes 60,000 images and labels.              |
     ----------------------------------------------------------------
    """
    def loadTestData(self):
        mnist = fetch_mldata('MNIST original')
        
        sets = np.arange(len(mnist.data))

        testIndex = np.arange(self.trainSize + 1, self.trainSize + self.testSize)

        images, labels = mnist.data[testIndex], mnist.target[testIndex]
        
        return images, labels
        
    """
     ----------------------------------------------------------------
    |    function: shuffleData()                                     |
    |    -----------------------                                     |
    |                                                                |
    |    Shuffles the images and labels in unison.                   |
     ----------------------------------------------------------------
    """
    def shuffleData(self, images, labels):
        s = np.arange(images.shape[0])
        np.random.shuffle(s)
        return images[s], labels[s]
        
    """
     ----------------------------------------------------------------
    |    function: convertLabels()                                   |
    |    -------------------------                                   |
    |                                                                |
    |    Converts labels to the neural network's representation of   |
    |    them. Example: 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]       |
     ----------------------------------------------------------------
    """
    def convertLabels(self, labels):
        encodedLabels = np.zeros((self.numOut, labels.shape[0]))
        for x, y in enumerate(labels):
            x = int(x)
            y = int(y)
            encodedLabels[y, x] = 1.0
        return encodedLabels
            
    """
     ----------------------------------------------------------------
    |    function: makeGuesses()                                     |
    |    -----------------------                                     |
    |                                                                |
    |    Converts an outputLayer (matrix of computed values) into    |
    |    a matrix of concrete guesses by deeming the neuron with     |
    |    the highest value as the guess for that image.              |
     ----------------------------------------------------------------
    """
    def makeGuesses(self, outputLayer):
        predictionMatrix = np.argmax(outputLayer, axis = 0)
        return predictionMatrix   
                 
    """
     ----------------------------------------------------------------
    |    function: calculateAccuracy()                               |
    |    -----------------------------                               |
    |                                                                |
    |    Given a matrix of guesses and a matrix of correct           |
    |    labels, this function calculates a percentage that          |
    |    represents the amount of correct guesses.                   |
     ----------------------------------------------------------------
    """
    def calculateAccuracy(self, predictions, labelMatrix, imageMatrix):
        accuracy = np.sum(labelMatrix == predictions, axis=0) / imageMatrix.shape[0]
        return accuracy
        
    """
     ----------------------------------------------------------------
    |    function: backPropagate()                                   |
    |    -------------------------                                   |
    |                                                                |
    |    Updates weight matrices using backpropagation.              |
     ----------------------------------------------------------------
    """
    def backPropagate(self, outputLayer, inputLayer, hiddenLayer, encodedSlice, learningRate):
        OutputNeuronCorrect = (1 - outputLayer)*(outputLayer)*(encodedSlice - outputLayer)
        HiddenNeuronCorrect = (hiddenLayer)*(1-hiddenLayer)*np.dot((self.weightMatrix2.T),OutputNeuronCorrect)
        InputNeuronCorrect = np.dot((self.weightMatrix1.T),HiddenNeuronCorrect)

        weightDifference1 = learningRate * HiddenNeuronCorrect.dot(inputLayer)
        weightDifference2 = learningRate * OutputNeuronCorrect.dot(hiddenLayer.T)
        
        newWeightMatrix1 = ((1- ((self.lambdaValue * learningRate)))*(self.weightMatrix1) + weightDifference1)
        newWeightMatrix2 = ((1- ((self.lambdaValue * learningRate)))*(self.weightMatrix2) + weightDifference2)
        
        return newWeightMatrix1, newWeightMatrix2
        
    """
     ----------------------------------------------------------------
    |    function: backPropagate2L()                                 |
    |    ---------------------------                                 |
    |                                                                |
    |    Updates weight matrices using backpropagation. Used for     |
    |    2-layered networks.                                         |
     ----------------------------------------------------------------
    """
    def backPropagate2L(self, outputLayer, inputLayer, hiddenLayer, hiddenLayer2, encodedSlice, learningRate):
        OutputNeuronCorrect = (1 - outputLayer)*(outputLayer)*(encodedSlice - outputLayer)
        HiddenNeuron2Correct = (hiddenLayer2)*(1-hiddenLayer2)*np.dot((self.weightMatrix3.T),OutputNeuronCorrect)
        HiddenNeuronCorrect = (hiddenLayer)*(1-hiddenLayer)*np.dot((self.weightMatrix2.T), HiddenNeuron2Correct)
        InputNeuronCorrect = np.dot((self.weightMatrix1.T),HiddenNeuronCorrect)

        weightDifference1 = learningRate*HiddenNeuronCorrect.dot(inputLayer)
        weightDifference2 = learningRate*HiddenNeuron2Correct.dot(hiddenLayer.T)
        weightDifference3 = learningRate*OutputNeuronCorrect.dot(hiddenLayer2.T)
        
        newWeightMatrix1 = ((1 - ((self.lambdaValue * learningRate))) * (self.weightMatrix1) + weightDifference1)
        newWeightMatrix2 = ((1 - ((self.lambdaValue * learningRate))) * (self.weightMatrix2) + weightDifference2)
        newWeightMatrix3 = ((1 - ((self.lambdaValue * learningRate))) * (self.weightMatrix3) + weightDifference3)
        
        return newWeightMatrix1, newWeightMatrix2, newWeightMatrix3
                
    """
     ----------------------------------------------------------------
    |    function: backPropagate3L()                                 |
    |    ---------------------------                                 |
    |                                                                |
    |    Updates weight matrices using backpropagation. Used for     |
    |    3-layered networks.                                         |
     ----------------------------------------------------------------
    """
    def backPropagate3L(self, outputLayer, inputLayer, hiddenLayer, hiddenLayer2, hiddenLayer3, encodedSlice, learningRate):
        OutputNeuronCorrect = (1 - outputLayer)*(outputLayer)*(encodedSlice - outputLayer)
        HiddenNeuron3Correct = (hiddenLayer3)*(1-hiddenLayer3)*np.dot((self.weightMatrix4.T),OutputNeuronCorrect)
        HiddenNeuron2Correct = (hiddenLayer2)*(1-hiddenLayer2)*np.dot((self.weightMatrix3.T),HiddenNeuron3Correct)
        HiddenNeuronCorrect = (hiddenLayer)*(1-hiddenLayer)*np.dot((self.weightMatrix2.T), HiddenNeuron2Correct)
        InputNeuronCorrect = np.dot((self.weightMatrix1.T),HiddenNeuronCorrect)

        weightDifference1 = learningRate*HiddenNeuronCorrect.dot(inputLayer)
        weightDifference2 = learningRate*HiddenNeuron2Correct.dot(hiddenLayer.T)
        weightDifference3 = learningRate*HiddenNeuron3Correct.dot(hiddenLayer2.T)
        weightDifference4 = learningRate*OutputNeuronCorrect.dot(hiddenLayer3.T)
        
        newWeightMatrix1 = ((1 - ((self.lambdaValue * learningRate))) * (self.weightMatrix1) + weightDifference1 )
        newWeightMatrix2 = ((1 - ((self.lambdaValue * learningRate))) * (self.weightMatrix2) + weightDifference2 )
        newWeightMatrix3 = ((1 - ((self.lambdaValue * learningRate))) * (self.weightMatrix3) + weightDifference3 )
        newWeightMatrix4 = ((1 - ((self.lambdaValue * learningRate))) * (self.weightMatrix4) + weightDifference4 )
        
        return newWeightMatrix1, newWeightMatrix2, newWeightMatrix3, newWeightMatrix4
        
    """
     ----------------------------------------------------------------
    |    function: learn()                                           |
    |    -----------------                                           |
    |                                                                |
    |    The main computing function of the neural network. Uses     |
    |    feed-forward and backpropagation iteratively to learn       |
    |    using training data.                                        |
     ----------------------------------------------------------------
    """
    def learn(self, images, labels):
        
        ### Choose random weights. ###
        if (self.numLayers == 1):
            self.weightMatrix1, self.weightMatrix2 = self.chooseWeights()
        elif (self.numLayers == 2):
            self.weightMatrix1, self.weightMatrix2, self.weightMatrix3 = self.chooseWeights()
        elif (self.numLayers == 3):
            self.weightMatrix1, self.weightMatrix2, self.weightMatrix3, self.weightMatrix4 = self.chooseWeights()
        
        for e in range(self.epochs):
            
            ### Epoch level. ###
            print("--- I'm learning ... %d/%d epoch." % (e, self.epochs))
            
            ### Shuffle the data before each iteration. ###
            images, labels = self.shuffleData(images, labels)
            encodedLabels = self.convertLabels(labels)
            
            ### K-Fold. ###
            fold = np.array_split(range(labels.shape[0]), self.foldSize)
            
            for f in fold:
                
                if (self.numLayers == 1):
                    ### FeedForward. ###
                    inputLayer, sum1, hiddenLayer, sum2, outputLayer = self.feedForward(images[f],
                                                                                        self.weightMatrix1,
                                                                                        self.weightMatrix2)
              
                    ### Backpropagation. ###
                    y = encodedLabels[:, f]
                    
                    new1, new2 = self.backPropagate(outputLayer, 
                                                    inputLayer, 
                                                    hiddenLayer, 
                                                    y, 
                                                    self.learningRate)
                    self.weightMatrix1 = new1
                    self.weightMatrix3 = new2
                    
                elif (self.numLayers == 2):
                    ### FeedForward. ###
                    inputLayer, sum1, hiddenLayer, sum2, hiddenLayer2, sum3, outputLayer = self.feedForwardML(images[f],
                                                                                        self.weightMatrix1,
                                                                                        self.weightMatrix2,
                                                                                        self.weightMatrix3,
                                                                                        self.weightMatrix4)
              
                    ### Backpropagation. ###
                    y = encodedLabels[:, f]
                    
                    new1, new2, new3 = self.backPropagate2L(outputLayer, 
                                                    inputLayer, 
                                                    hiddenLayer,
                                                    hiddenLayer2, 
                                                    y, 
                                                    self.learningRate)
                    self.weightMatrix1 = new1
                    self.weightMatrix2 = new2
                    self.weightMatrix3 = new3
                                        
                elif (self.numLayers == 3):
                    ### FeedForward. ###
                    inputLayer, sum1, hiddenLayer, sum2, hiddenLayer2, sum3, hiddenLayer3, sum4, outputLayer = self.feedForwardML(images[f],
                                                                                        self.weightMatrix1,
                                                                                        self.weightMatrix2,
                                                                                        self.weightMatrix3,
                                                                                        self.weightMatrix4)
              
                    ### Backpropagation. ###
                    y = encodedLabels[:, f]
                    
                    new1, new2, new3, new4 = self.backPropagate3L(outputLayer, 
                                                    inputLayer, 
                                                    hiddenLayer,
                                                    hiddenLayer2,
                                                    hiddenLayer3, 
                                                    y, 
                                                    self.learningRate)
                    self.weightMatrix1 = new1
                    self.weightMatrix2 = new2
                    self.weightMatrix3 = new3
                    self.weightMatrix4 = new4
    
    """
     ----------------------------------------------------------------
    |    function: test()                                            |
    |    ----------------                                            |
    |                                                                |
    |    Function that takes given training data and uses learned    |
    |    weights to determine what they are.                         |
     ----------------------------------------------------------------
    """
    def test(self, images, labels):

        if (self.numLayers == 1):
            ### Feed forward once, try to guess what the images are. ###
            inputLayer, sum1, hiddenLayer, sum2, outputLayer = self.feedForward(images, 
                                                                                self.weightMatrix1, 
                                                                                self.weightMatrix2)
            predictions = self.makeGuesses(sum2)
            return predictions
            
        if (self.numLayers == 2):
            ### Feed forward once, try to guess what the images are. ###
            inputLayer, sum1, hiddenLayer, sum2, hiddenLayer2, sum3,  outputLayer = self.feedForwardML(images, 
                                                                                                       self.weightMatrix1, 
                                                                                                       self.weightMatrix2,
                                                                                                       self.weightMatrix3,
                                                                                                       self.weightMatrix4)
            predictions = self.makeGuesses(sum3)
            return predictions
            
        if (self.numLayers == 3):
            ### Feed forward once, try to guess what the images are. ###
            inputLayer, sum1, hiddenLayer, sum2, hiddenLayer2, sum3, hiddenLayer3, sum4, outputLayer = self.feedForwardML(images, 
                                                                                                                          self.weightMatrix1, 
                                                                                                                          self.weightMatrix2,
                                                                                                                          self.weightMatrix3,
                                                                                                                          self.weightMatrix4)
            predictions = self.makeGuesses(sum4)
            return predictions
        
        
"""
 ----------------------------------------------------------------
|    main()                                                      |
|    ------                                                      |
|                                                                |
|    Where the magic happens.                                    |
 ----------------------------------------------------------------
"""
def main():
    neuralNetwork = NeuralNetwork(numOut = 10,
                                  numLayers = 1, 
                                  numHiddenNeurons = 150,
                                  epochs = 100,
                                  foldSize = 50,
                                  trainSize = 60000,
                                  testSize = 10000,
                                  learningRate = 0.001,
                                  lambdaValue = 0.1)
                                  
    print("\n==========================================================\n")
    print("Initializing Neural Network with the following attributes:\n")
    print("{ Layers: %d, Hidden Neurons: %d\n"
          "Size of training data: %d\n"  
          "Size of testing data: %d\n"  
          "Fold Sizes: %d, Learning Rate: %.3f\n"
          "Lambda Value: %.3f }\n" 
         % (neuralNetwork.numLayers, 
            neuralNetwork.numHiddenNeurons,
            neuralNetwork.trainSize,
            neuralNetwork.testSize,
            neuralNetwork.foldSize, 
            neuralNetwork.learningRate, 
            neuralNetwork.lambdaValue) )
    
    print("Starting timer...\n")
    start = datetime.now()
                                  
    ### Training ###
    print(" -------------- ")
    print("|              |")
    print("|   Training   |")
    print("|              |")
    print(" -------------- ")


    # Loading train data.
    featureSet, labelSet = neuralNetwork.loadTrainData()

    # Train.
    neuralNetwork.learn(featureSet, labelSet)
             
    ### Testing ###
    print(" -------------- ")
    print("|              |")
    print("|   Testing    |")
    print("|              |")
    print(" -------------- ")
    
    # Loading test data.
    featureSet, labelSet = neuralNetwork.loadTestData()
    
    # Test our network.
    predictions = neuralNetwork.test(featureSet, labelSet)
    
    # Calculate accuracy.
    accuracy = neuralNetwork.calculateAccuracy(predictions, labelSet, featureSet)

    print("Training accuracy: %.2f%%\n" % (accuracy * 100))
    
    print("Finishing...\n")
    end = datetime.now()

    print("Finished in {}".format(end-start))
    
    print("\n==========================================================\n")
    
"""
 --------------
|              |
|    Start.    |
|              |
 --------------
"""
main()
