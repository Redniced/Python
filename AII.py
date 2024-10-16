import numpy as np
print(np.__version__)

  #X = input of ouyr 3 input in the XOR gate
  #Next, we define all eight possibilities of our X1–X3 inputs and the Y1 output from
X = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,0],[1,0,1],[1,1,0],[1,1,1],), dtype=float)
  #y = the output from the nuaral network - y1
y = np.array(([1],[0],[0],[0],[0],[0],[0],[0],[1],), dtype=float)
  #We now choose a value to predict. We will predict them all, but the value stored in xPredicted will be the answer that we want at the end:
  #xpredicted, what value we want in the end
xPredicted = np.array(([1 ,1,1]), dtype=float)
  #Caluclates the maximum input for X which is [1,1,1] and normalises the date for the pc to understand the outcome
X = X / np.amax(X, axis=0) 
print(X)
  #same here, but for the xPredicted value
xPredicted = xPredicted / np.amax(xPredicted, axis=0)
print(xPredicted)
  #Save the Sum Squared Loss results to a file (csv) per epoch(try):
lossFile = open("SumSquaredLossList.csv", "w")

#The class for the nutral network we are making
class Neural_Network (object): 
    def __init__(self): 
        # parameters 
        self.inputLayerSize = 3  # X1,X2,X3 
        self.outputLayerSize = 1 # Y1 
        self.hiddenLayerSize = 4 # Size of the hidden layer
        #Set all the network weights to random values initially
        # 3x4 matrix for input to hidden 
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) 
        # 4x1 matrix for hidden layer to output 
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
    def feedForward(self, X): 
        # Conecting the input-layr to the hidden-layer, by X * W1, Z is the weighted sum for the hidden layer neurons.
        self.z = np.dot(X, self.W1)
        
        # the activationSigmoid activation function - to calculate the "activation" of the hidden layer neurons fire or not
        self.z2 = self.activationSigmoid(self.z) 
        
        # Is calucalted by the second set of weight to produce the weighted sum of the out put layer
        self.z3 = np.dot(self.z2, self.W2) 
        
        # final activation function - fire or not
        o = self.activationSigmoid(self.z3) 
        return o
    def backwardPropagate(self, X, y, o): 
        # y is the actual output and o is the predicted output from the feedforwards, o_error tells us the adjusted output error
        self.o_error = y - o 
 
        # activationsigmoindPrime = derivative of the sigmoind active function, derivative * error = how sensetive the output is to changes in the weighted sum
        self.o_delta = self.o_error * self.activationSigmoidPrime(o) 
 
        #messure how the weights in from the hiddenm layer to output layr affected the  error in the output o
        self.z2_error = self.o_delta.dot(self.W2.T) 
 
        # applying derivative of activationSigmoid to z2 error, showing how much each hidden layer contributed to the overall error and how much we need to adjust the weights 
        self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2) 
 
        # adjusting first set (inputLayer --> hiddenLayer) weights
        #X.T (transpose) dimensions match for the matrix multiplication, the dot product adjust the hidden layr product
        self.W1 += X.T.dot(self.z2_delta) 
        # adjusting second set (hiddenLayer --> outputLayer) weights 
        self.W2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, X, y): 
        # feed forward the loop 
        o = self.feedForward(X) 
        # and then back propagate the values (feedback) 
        self.backwardPropagate(X, y, o)

    def activationSigmoid(self, s): 
        # activation function 
        # s = weighted sum of neuron inputs and converts it between 0 and 1, to control the ouput of nurons
        return 1 / (1 + np.exp(-s)) 
 
    def activationSigmoidPrime(self, s):
        # First derivative of activationSigmoid 
        # adjusting the weights in proportion to how sensetive outputs are to cahnge inputs
        return s * (1 - s)
        #Saving the epoch values to the loss function to the lost file
    
    def saveSumSquaredLossList(self,i,error): 
      lossFile.write(str(i) + "," + str(error.tolist()) + '\n') 
 
    def saveWeights(self): 
      # save this in order to reproduce our cool network 
      np.savetxt("weightsLayer1.txt", self.W1, fmt="%s") 
      np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")
    
    def predictOutput(self): 
        print ("Predicted XOR output data based on trained weights: ") 
        print ("Expected (X1-X3): \n" + str(xPredicted)) 
        print ("Output (Y1): \n" + str(self.feedForward(xPredicted))) 
 
myNeuralNetwork = Neural_Network() 
trainingEpochs = 1000

#Following is the main training loop that goes through all requested epochs. Change the trainingEpochs variable in the preceding code snippet to vary the num- ber of epochs you would like to train your network

for i in range(trainingEpochs): # train myNeuralNetwork 1,000 times 
    print ("Epoch # " + str(i) + "\n") 
    print ("Network Input : \n" + str(X)) 
    print ("Expected Output of XOR Gate Neural Network: \n" + str(y)) 
    print ("Actual Output from XOR Gate Neural Network: \n" + str(myNeuralNetwork.feedForward(X))) 
    # mean sum squared loss 
    Loss = np.mean(np.square(y - myNeuralNetwork.feedForward(X))) 
    myNeuralNetwork.saveSumSquaredLossList(i,Loss) 
    print ("Sum Squared Loss: \n" + str(Loss)) 
    print ("\n") 
    myNeuralNetwork.trainNetwork(X, y)

myNeuralNetwork.predictOutput()