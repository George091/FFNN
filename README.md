# FFNN

## GENERAL OVERVIEW

This is the readme for the "FFNN.py" file, created by George Barker for CSCI315 Artificial Intelligence with Professor Cody Watson. 

This python file codes a FEED FORWARD NEURAL NETWORK (FFNN) with 97% accuracy trained using the MNIST dataset to recognize and predict handwritten digits.


## HOW TO RUN CODE

To run the program and see the model's performance: Run the python file "FFNN.py". Given all used packages are installed, a model serialized using pickle will be imported from the file "FFNNs". The model contained in this file has an accuracy of 97%. Using this method allows you to skip the training process to see what this model is capable of. The model's accuracy will be tested on data not trained on, which is 20% of the MNIST dataset, and 10 of my own hand drawn images (computer drawn using GIMP). The network I've trained is a model with 800 nodes for the first hidden layer, 500 nodes for the second hidden layer, and a learning rate of 0.01. 

To watch a new model train using 80% of the MNIST dataset: Uncomment lines 187 to 195 and run the python file "FFNN.py". The model will be tested before each epoch of the training data. During testing, the model will attempt to predict the handwritten digit for 14,000 of the handwritten digits in the MNIST test set. For wrong predictions, the model prints to the python console what the actual handwritten digit is, the model's prediction and it's confidence in that prediction, and the raw one hot encoded model's prediction. * note it is not necessary to comment out the pickle code that brings in the serialized FFNN as a new FFNN is initialized over that serialized FFNN.

## METHODS

### EXTRACTING DATA:

I downloaded 70,000 28x28 images of handwritten digits 0-9 from the MNIST database. To do so, I used the mnist.load_data() function, which returns 2 tuples as (x_train, y_train) and (x_test, y_test). x_train and x_test is an array of grayscale image data with shape (num_samples,28,28). y_train and y_test are an array of digit labels in range 0-9 with shape (num_samples,). x_train for each image has a 28x28 array with RGB values filled in. y_train has the label at the index for what the handwritten digit actually is. I then concatenated the x_train and x_test arrays, as well as the y_train and y_test arrays. I broke up the data into 60% training set, 20% validation set, and 20% for the test set. In practice I ended up using the 60% training set and 20% validation set as one epoch to train the model. I found that overfitting on the MNIST dataset did not happen in the 5 epochs I was performing, thus I found it more beneficial to use the extra 20% from the validation set to train on. The testing set was not used to train on and was saved for testing.

To input the data into the network I flattened the 28 x 28 array for each image into a 1D array. This entailed taking the 28 x 28 array of pixels for each image and reshaping it into a 784x1 array. With this array, each pixel in the array can be input into the network as one node if we have an input layer of size 784. This makes it easy to feedforward our input into the neural network, as we simply consider each pixel value as input for each node in the input layer. This allows each pixel to be accounted for as a feature. I also divided each pixel value by 255 in the input vector. This standardizes values between 0 and 1 for the input layer. This is beneficial as it prevents inputs from being extremely large or extremely small as is the case with pixel values of either 0 or 255. When the input values are on a similar scale, we can reduce the likelihood of weights encountering exploding or vanishing gradients. 

Finally, I converted the Y values from MNIST into a one hot vector. This will allow the model to compare network predictions to actual values (called the target) during back propagation. It is important to note, that during back propagation I actually made the target of 1 -> .99 and the target of 0 -> .01. Since we use the sigmoid activation function, which gives values between 0 and 1, sigmoid can't actually give values of 0 and 1. Therefore, changing the target in such a manner allows the model to "hit" the target and allows the derivative of error to actually become 0. This allows the model to reach the target and avoid the vanishing gradient problem, which occurs when the model continually changes the weights in training to reach the target of 1, but is never actually able to reach the target due to the use of the sigmoid activation function.

### INITIALIZATION

The FFNN is initialized with 3 weight matrices. One weight matrix connects the input and first hidden layer. Another connects the first hidden layer and second hidden layer. Lastly, one connects the second hidden layer and output layer. Weight matrices are made up of weights for each neuron to neuron connection. This is a fully connected network, in which every neuron in the previous layer is connected to every neuron in the next layer. Thus there is a weight associated with each connection. The first column of the weight matrix contains the connection from the first neuron in the first layer to every neuron in the second layer. Therefore, the weight matrices has a number of rows corresponding to the number of neurons in the second layer. The next column will contain the connection from the second neuron to every neuron in the next layer. Thus there is a column for every neuron in the first layer. Thus our weight matrix is (size of second layer)x(size of first layer).

I initialize the weights of the weight matrices using a normal distribution with mean of 0 and standard deviation as a factor of the number of nodes to the power of -0.5. We use a normal distribution because we want to initial weights to be close to 0. Setting the weights to 0 would be problematic as the model wouldn't be able to tell where the error in its prediction came from, and it would not allow us to effectively tune the network's weights. On the other hand if we randomly assign values without a normal distribution, and get weights far away from each other, this may make it difficult for the network to change the weights effectively as all the weights are far apart. It may take a long time for some weights to be tuned properly. On the other hand, when the weights are close to 0, they can all be changed effectively compared to one another. Normal distribution allows the weights to be set in a way that strikes this moderation between starting weights too far away from each other, and too close to each other. Finally we set the standard deviation proportional to the size of the hidden layer nodes, so that the weights can be further from the mean of 0 when there are more nodes in the layer.

### FEEDFORWARD

To find the value of the first hidden layer, I took the dot product of the input layer weight matrix and the input, respectively. Since the weight matrix row has the same output neuron, matrix multiplication in this manner returns a matrix for the summation of weights and input for each node in the first hidden layer. I used the sigmoid activation function to change the output for each of these nodes to give the first hidden layer output. I repeated the same matrix multiplication with the weight matrix between the first and second hidden layer to get the weighted input for the second layer. I then used the sigmoid activation again to get the second hidden layer output value. I use matrix multiplication between the weights matrix between the second hidden layer and output to find the weight input for the output layer. I then used the sigmoid activation on this layer. I implemented the Softmax output layer by dividing each output by the sum of the outputs for the output layer. This means all the nodes in the output layer add to 1. Since the output can be one of 10 digits 0 - 9, there are 10 nodes in the output layer. Each node is treated as 0-9 respectively, from first node in the output to last node in the output. Feedforward results in some prediction or probability for each node, and the max prediction can be used as a probability for a prediction for classification of a digit. Importantly, the sigmoid activation function adds non-linearity to our model, allowing the model to find a relationship between variables that is more complex. This non-linearity allows our model to map input to predictions more accurately.

### BACK-PROPAGATION AND GRADIENT DESCENT

Optimally, in the output I want the model to predict a correct digit with a value close to 1 and an incorrect digit with a value close to 0. During training, the model gives a certain output or prediction for each value based off its weights, and so based off the network's output we tune the weights such that next time the value of the prediction will be closer to 1 for the correct digit and closer to 0 for the incorrect digits. I used gradient descent to change the weight matrices by subtracting the derivative of error with respect to weights from the current weight matrices. Using gradient descent here makes sense because we can find the derivative of error with respect to each weight for every neuron, therefore we can multiply this derivative by a learning rate and iteratively reduce our error.

I considered error as error = target - output, where target is 1 or 0 depending on whether the digit is correct or incorrect; and the output is the networks prediction. In data extraction, I explained I converted the MNIST Y digits to a one hot vector (I also explained why .99 and .01 were used as targets). This means error for a prediction can be simply calculated as Y_actual - Y or (target - prediction). This gives the error for the output layer. 

I then calculated an error matrix for each hidden layer. This error matrix contains the amount of error that each node in the layer is responsible for. I found the error matrix for the second hidden layer by taking the transpose of the 2nd hidden layer to output layer's weight matrix and dot producting that with the error of the output layer. This matrix multiplication returns the error matrix for the second hidden layer because each row in the transposed weight matrix contains the nodes headed from one unique node in the hidden layer to every node in the output layer, and multiplication with the error matrix of the output layer returns the summation of the error for each weight leading to a unique node in the previous layer. Therefore, the error matrix contains the magnitude of error each node is responsible for. The logic and assumption here is that I assume larger weights between two nodes contribute more to error. To find the error matrix for the first hidden layer I calculated the dot product of the transpose of (the weight matrix between the first and second hidden layer) and the error matrix for the second hidden layer. This calculation results in the error matrix for the first hidden layer. There is no need to calculate any error matrix for the first layer, as the first layer is the input. Thus far I have explained the calculation for how I found how much error each node is responsible for in the first hidden layer, second hidden layer, and output layer.

To find the change in error with respect to change in weights for the output layer, I used the chain rule to break this derivative up into derivatives we do know. According to the chain rule, (the derivative of error with respect to weight) is equal to (the derivate of error with respect to output) times (the derivative of output with respect to input) times (the derivative of input with respect to weights). In my FFNN model's train function, the change in error with respect to change in output is simply the derivative of our error function, which is -(t-o). The derivative of output with respect to input is the derivate of the weighted input of our output layer going into our sigmoid activation function. The derivative of output with respect to input is the derivative of sigmoid(x), or sigmoid(x)*sigmoid(1-x), where x is weighted input of the output layer. I used this as the derivative as the transition from weighted input to output for the output layer (via softmax layer) uses the sigmoid function, therefore it makes sense to use the derivative of sigmoid. 

In my code the elementwise multiplication of the first two terms of the chain rule are referred to as outputDelta. outputDelta thus represents derivative of change in error with respect to the 1st input node (for the first element), all the way down to the change in error with respect to the last node in the ouput layer input. The derivate of input with respect to weights is simply going to be the output of the second hidden layer. This is because the summation input of a neuron is weight * input of previous layer, and the derivative of this with respect to weight is simply the input from the previous layer. Thus I dot product the outputDelta variable, which is derivative of error with respect to input, with the transpose of the second hidden layer's output (aka derivative of input with respect to weight for the output layer) to get derivative of error with respect to weight for the output layer. I used the dot product as the derivative of input with respect to weight for weight 1->1 is the same as 1->x (where x is any node in the output layer). The derivative with respect to weight for any weight coming from neuron 1 in the second hidden layer is the same for all weights starting at neuron 1 in the second hidden layer. This derivative does not change if the neuron goes from the same output node to the first output node or the 10th output node. The derivative only depends on the output of the previous layer. Therefore, the dot product of the outputDelta variable and second hidden layer's output gives the derivative of error with respect to weights for the output layer. The resulting matrix is used as the derivative in the gradient descent algorithm to adjust the weight matrix and can be multiplied by a scalar learning rate. Since we calculated the derivative of error with respect to weights for the output layer, we can only use this matrix to update the weights between the second hidden layer and the ouput layer. We must do backpropagation using the chain rule to find the derivative of error with respect to weight for the other weight matrices.

I used back propagation to find the change in error with respect to weight at each of the hidden layers in a similar way to the above output layer. I will recap this for the hidden layers. The hidden layers each have an error matrix. This gives the amount of error that the layer is responsible for. We use the chain rule the exact same way as above, taking the derivative of error with respect to weights as (the derivate of error with respect to output) times (the derivative of output with respect to input) times (the derivative of input with respect to weights). The mathematical formula according to my FFNN model to backpropagate is dE/dW = -(t-ok)*sigmoid derivative(weighted input of layer k)*oj where j is the layer occuring before k. Again, (t-ok) = the previously calculated error matrix for the layer and plugging into the error derivative by multiplying by -1 gives the derivative of error with respect to output. We must again find the derivative of output with resepect to the input (accounting for the modification of error by our sigmoid activation function). Again I wrote a function finding the derivative of sigmoid ( which is sigmoid(x)*sigmoid(1-x) given an input, thus I put my summation for the second hidden layer into this function to find the derivative of the output with respect to input for the second hidden layer. Thus for the second hidden layer, I took the second layer's error matrix and multiplied elementwise by the sigmoid derivative of weighted summation into hidden layer 2. I then dot producted this resulting matrix with the transpose of my activated output for hidden layer 1 to get dE/dW to update the weight matrix between the first and second hidden layer. Similarily for the first hidden layer, I took the first layer's error matrix, multiplied elementwise by the sigmoid derivative of weighted summation into hidden layer 1. I then dot producted this resulting matrix with the transpose of my input layer to get dE/dW to update the weight matrix between the input and first hidden layer. I  merely recap what I did here as the logic behind why I used transpose, dot product, or element wise multiplication is all previously explained for the output layer. The logic here is the same.

### PERFORMANCE

My model serialized in the file "FFNNs" using pickle contains 800 nodes for the first hidden layer, 500 nodes for the second hidden layer, and a learning rate of .01. It is important to note some limitations of my model. When flattening a 28x28 image to a 784x1 array, we lose the ability to give the positional context of a pixel value relative to other pixel values to the FFNN. Error can come from not having this positional context. Convolutional Neural Networks (CNN) use a convolutional layer that can more adequately hold on to this positional information. An additional note on the accuracy of the model is the classification of GIMP images. The GIMP images for my stored model has an accuracy of 100%. However, on other trained models I found it likely for this number to fluctuate between 80-100%. A reason for this potential discrepency between accuracy on other trained models is that the GIMP images were not normalized exactly the same way as the Keras dataset. Although I imported 28x28 images, flattened the vector to 784x1, and normalized the pixel values to 0-1, there are other potential sources of error differentiating my GIMP images from that of the Keras dataset. Potential sources of error include centering of the handwritten digit, the brush stroke size used to make the digit, or use of a computer mouse to draw the image (versus the technique used to draw those of the Keras dataset). Given these limitations, my model performed at 97% accuracy on the Keras test set, and 100% on my GIMP image dataset. 

## Sources
https://keras.io/datasets/
https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
https://enlight.nyc/projects/neural-network/
https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3
https://www.youtube.com/watch?v=UJwK6jAStmg
http://neuralnetworksanddeeplearning.com/chap1.html
http://neuralnetworksanddeeplearning.com/chap2.html
https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html
https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python

