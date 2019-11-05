# -*- coding: utf-8 -*-
"""
Author: George Barker
Date: October 10, 2019

Feed-Forward Neural Network that predicts handwritten digits. Takes in 28x28
grayscale images from the MNIST database, trains a neural network, and tests 
the model on prediction of handwritten digits. Model saved using pickle has
97.01% accuracy on testing data.

This FFNN contains an input layer, 2 hidden layers, and an output layer. Both 
hidden layers use a sigmoid activation function, and the output is a softmax
layer.

"""
import numpy as np 
import matplotlib
import keras
import pickle

class NeuralNetwork:
    def __init__(self, hiddenSize1 = 800, hiddenSize2 = 500, learningRate = .01):    
        # hyperparameters
        self.hiddenLayer1Size = hiddenSize1
        self.hiddenLayer2Size = hiddenSize2
        self.learningRate = learningRate       
        self.inputLayerSize = 784
        self.outputLayerSize = 10
        # weights
        self.inputToHidden1Weights = self.initWeightMatrix(self.inputLayerSize,self.hiddenLayer1Size)
        self.hidden1ToHidden2Weights = self.initWeightMatrix(self.hiddenLayer1Size,self.hiddenLayer2Size)
        self.hidden2ToOutputWeights = self.initWeightMatrix(self.hiddenLayer2Size,self.outputLayerSize)
       
    def initWeightMatrix(self, firstLayerSize,secondLayerSize):
        """Returns a matrix containing randomly initialized weights"""
        return np.array([[np.random.normal(0, secondLayerSize**(-.5)) for i in range(firstLayerSize)] for j in range(secondLayerSize)])
    
    def train(self, X, Y_actual):
        """ Given X as a 784 array and Y as a one-Hot size 10 array 
        feedforward and backpropagate to update weight matrix """
        X = np.array(X).reshape(-1,1)
        Y_actual = np.array(makeTarget(Y_actual)).reshape(-1,1)
        ######################## FORWARD PASS ########################
        # Sigmoid - 1st hidden layer
        Zh1 = np.dot(self.inputToHidden1Weights, X)
        Ah1 = sigmoid(Zh1)
        # Sigmoid - 2nd hidden layer
        Zh2 = np.dot(self.hidden1ToHidden2Weights, Ah1)
        Ah2 = sigmoid(Zh2)
        # SoftMax - output
        Zo = np.dot(self.hidden2ToOutputWeights, Ah2)
        Y = np.array(softmax(Zo))
        ######################## BACKWARD PASS ########################
        ## Error Matrices ##
        outputError = -(Y_actual - Y)
        secondHiddenError = np.dot(self.hidden2ToOutputWeights.transpose(), outputError)
        firstHiddenError = np.dot(self.hidden1ToHidden2Weights.transpose(), secondHiddenError)
        outputDelta = -(Y_actual - Y)*sigmoid_prime(Zo)
        secondHiddenDelta = secondHiddenError * sigmoid_prime(Zh2)
        firstHiddenDelta = firstHiddenError * sigmoid_prime(Zh1)
        ## OUTPUT LAYER TO HIDDEN LAYER 2 ##
        outputLayerDeltaErrorWithRespectToWeight = np.dot(outputDelta,Ah2.transpose())
        ## HIDDEN LAYER 2 TO HIDDEN LAYER 1 ##
        HiddenLayer2DeltaErrorWithRespectToWeight = np.dot(secondHiddenDelta,Ah1.transpose())
        ## HIDDEN LAYER 1 TO INPUT LAYER ## 
        HiddenLayer1DeltaErrorWithRespectToWeight = np.dot(firstHiddenDelta,X.transpose())
        ######################## GRADIENT DESCENT ########################
        self.hidden2ToOutputWeights -= self.learningRate * (outputLayerDeltaErrorWithRespectToWeight)
        self.hidden1ToHidden2Weights -= self.learningRate * (HiddenLayer2DeltaErrorWithRespectToWeight)
        self.inputToHidden1Weights -= self.learningRate * (HiddenLayer1DeltaErrorWithRespectToWeight)
        
    def predict(self, X):
        """ Returns a tuple of [0] the predicted digited, [1] the confidence
        and [2] the raw one hot encoded Y prediction """
        X = np.array(X).reshape(-1,1)
        ######################## FORWARD PASS ########################
        # Sigmoid - 1st hidden layer
        Zh1 = np.dot(self.inputToHidden1Weights, X)
        Ah1 = sigmoid(Zh1)
        # Sigmoid - 2nd hidden layer
        Zh2 = np.dot(self.hidden1ToHidden2Weights, Ah1)
        Ah2 = sigmoid(Zh2)
        # SoftMax - output
        Zo = np.dot(self.hidden2ToOutputWeights, Ah2)
        Y = np.array(softmax(Zo))
        prediction = (reverseOneHot(Y),round(np.amax(Y),3), Y)
        return prediction
        
    def test(self, X, Y):
        """ Tests the model on the given input """
        correct = 0
        total = 0
        for i in range(len(X)):
            prediction = self.predict(X[i])
#            print("###################### ITERATION #: ",i," ######################")
#            print("Actual is: "+str(reverseOneHot(Y[i]))+"; Network prediction: "+str(prediction))
            if (prediction[0] == reverseOneHot(Y[i])):
                correct += 1
            else:
                print("Incorrect prediction "+str(total-correct)+": Actual is: "+str(reverseOneHot(Y[i]))+"; Network prediction: "+str(prediction[:2])+", raw one hot prediction: \n"+str(prediction[2]))
            total+=1
        print('Accuracy of the FFNN on '+str(len(X))+' test images: '+str(round((100 * correct / total),2))+'%')
        
def makeTarget(numpyArrayHot):
    """ Updates One Hot Encoded vector to target of .99 instead of 1 and
    .01 instead of 0 to eliminate vanishing gradient due to use of sigmoid
    activation function """
    numpyArray = numpyArrayHot.copy()        
    numpyArray[np.where(numpyArrayHot==np.max(numpyArrayHot))] = .99
    numpyArray[np.where(numpyArrayHot!=np.max(numpyArrayHot))] = .01
    return numpyArray

def reverseOneHot(numpyArrayHot):
    """ Takes in a one hot array and returns the index of one """
    numpyArray = numpyArrayHot.copy()        
    numpyArray[np.where(numpyArrayHot==np.max(numpyArrayHot))] = 1
    numpyArray[np.where(numpyArrayHot!=np.max(numpyArrayHot))] = 0
    return np.asscalar(np.where(numpyArray==1)[0])

def sigmoid(x):
    """Apply sigmoid activation function"""
    return 1/(1+np.exp(-1*x))

def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    """Apply softmax activation function"""
    return sigmoid(x) * (1/sum(sigmoid(x)))

def loadData():
    """ Returns an array of 60% training, 20% validation, and 20% test set 
    as tuples from the MNIST database"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    num_classes = 10
    train_size = .6
    valid_size = .2
    test_size = .2
    image_vector_size = 28*28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x = np.concatenate((x_train, x_test))
    x = x/255
    y = np.concatenate((y_train, y_test))
    train_index=int(len(x)*train_size)
    valid_index=train_index+int(len(x)*valid_size)
    test_index=valid_index+int(len(x)*test_size)
    (x_train, y_train) = (x[:train_index], y[:train_index])
    (x_valid, y_valid) = (x[train_index:valid_index], y[train_index:valid_index])
    (x_test, y_test) = (x[valid_index:test_index], y[valid_index:test_index])
    return [(x_train, y_train),(x_valid, y_valid),(x_test, y_test)]
    
def getPixelArray(filename):
    return np.array(matplotlib.pyplot.imread(filename).reshape(28*28,1))

def loadData_GIMP():
    """ Load flattened arrays of greyscale 28 x 28 png handwritten digits """
    GIMP_X = []
    GIMP_Y = []
    for index in range(10):
        pixelArray = getPixelArray("GIMPdigits/"+str(index)+".png")
        GIMP_X.append(pixelArray)
        GIMP_Y_One_Hot = ([0]*10)
        GIMP_Y_One_Hot[index] = 1
        GIMP_Y.append(GIMP_Y_One_Hot)
    GIMP_X = np.asarray(GIMP_X)
    GIMP_Y = np.asarray(GIMP_Y)
    return (GIMP_X,GIMP_Y)

# Load training, validation, and test data; and images made using GIMP
dataMNIST = loadData()
(x_train, y_train) = dataMNIST[0]
(x_valid, y_valid) = dataMNIST[1]
(x_test, y_test) = dataMNIST[2]
dataGIMP = loadData_GIMP()
GIMP_X = dataGIMP[0]
GIMP_Y = dataGIMP[1]

# Load the Neural Network
pickle_in = open("FFNNs","rb")
NN = pickle.load(pickle_in)

## Train model 
#NN = NeuralNetwork(800, 500, .01)
#print("Training Begins:")
#for epoch in range(4):
#    NN.test(x_test, y_test)
#    print("Epoch number: "+str(epoch)+", continuing to train...")
#    for index in range(42000):
#        NN.train(x_train[index],y_train[index])
#    for index in range(14000):
#        NN.train(x_valid[index],y_valid[index])

# Test model
print("Final test:")
NN.test(x_test, y_test)
print("GIMP test:")
NN.test(GIMP_X,GIMP_Y)

#pickle_out = open("FFNNs","wb")
#pickle.dump(NN, pickle_out)
#pickle_out.close()



