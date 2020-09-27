#####################################################################################################################
#   Assignment 2: Neural Network Programming
#   This is a starter code in Python 3.6 for a 1-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   NeuralNet class init method takes file path as parameter and splits it into train and test part
#         - it assumes that the last column will the label (output) column
#   h - number of neurons in the hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   W_hidden - weight matrix connecting input to hidden layer
#   Wb_hidden - bias matrix for the hidden layer
#   W_output - weight matrix connecting hidden layer to output layer
#   Wb_output - bias matrix connecting hidden layer to output layer
#   deltaOut - delta for output unit (see slides for definition)
#   deltaHidden - delta for hidden unit (see slides for definition)
#   other symbols have self-explanatory meaning
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################
# James Hooper ~ NETID: jah171230
# Hritik Panchasara ~ NETID: hhp160130

import numpy as np
import pandas as pd
from random import seed, randint
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, dataFile, activation, state, randseed, h1, h2, testsize, header=True):
        self.activation = activation
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h represents the number of neurons in the hidden layer
        #raw_input = pd.read_csv(dataFile)
        #url = "https://raw.githubusercontent.com/jamesH-48/Neural-Network-A2/master/smalltest.csv"
        #url = "https://raw.githubusercontent.com/jamesH-48/Neural-Network-A2/master/EEG%20Eye%20State.csv"
        url = "https://raw.githubusercontent.com/jamesH-48/Neural-Network-A2/master/balance-scale-1.csv"
        raw_input = pd.read_csv(url, header=0)
        # TODO: Remember to implement the preprocess method
        proc_X, proc_y = self.preprocess(raw_input)

        self.Xdf = proc_X
        self.Xnp = self.Xdf.to_numpy()
        self.Ydf = proc_y
        self.Ynp = self.Ydf.to_numpy()

        self.Xnp = self.Xnp.astype(float)
        self.Ynp = self.Ynp.astype(float)

        self.X, self.Xtest, self.y, self.ytest = train_test_split(self.Xnp,self.Ynp,test_size=testsize,random_state=state)

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[1])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])

        # Let's use a seed for the random values to track the exact input
        np.random.seed(randseed)

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden1 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.Wb_hidden1 = 2 * np.random.random((1, h1)) - 1

        self.W_hidden2 = 2 * np.random.random((h1, h2)) - 1
        self.Wb_hidden2 = 2 * np.random.random((1, h2)) - 1

        self.W_output = 2 * np.random.random((h2, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        self.deltaHidden1 = np.zeros((h1, 1))
        self.deltaHidden2 = np.zeros((h2, 1))
        self.h1 = h1
        self.h2 = h2

    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "relu":
            self.__relu(self, x)
        if activation == "tanh":
            self.__tanh(self, x)

    #
    # TODO: Define the derivative function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "relu":
            self.__relu_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __relu(self, x):
        return x * (x > 0)

    def __relu_derivative(self, x):
        return 1 * (x > 0)

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __tanh_derivative(self, x):
        return 1 - (x * x)

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, data):
        newdataX = data[["LW","LD","RW","RD"]]
        newdataY = data[["Class"]]
        columnsY = ['L','B','R']
        y_df = pd.DataFrame(columns=columnsY)
        y_df.loc[0] = (0,0,1)
        for i in range(len(newdataY)):
            if newdataY.Class[i] == 'L':
                y_df.loc[i] = (1,0,0)
            if newdataY.Class[i] == 'B':
                y_df.loc[i] = (0, 1, 0)
            if newdataY.Class[i] == 'R':
                y_df.loc[i] = (0, 0, 1)

        return newdataX, y_df

    # Below is the training function

    def train(self, max_iterations, learning_rate):
        # We don't use learning rate from the function

        # Adam Optimizer Variables
        # Recommended values: alpha = 0.001, beta1 = 0.9, beta2 = 0.999 and epsilon = 10**−8
        alpha = .001
        beta1 = .9
        beta2 = .999
        epsilon = 10 ** -8
        # Array variables for each gradient
        # Gradient for output, hidden layer 2, and hidden layer 1 weights
        # Includes biases for each
        # m[0] & v[0] are for update_weights_output
        # ....
        # m[5] & v[5] are for update_weight_hidden_b
        m = [0, 0, 0, 0, 0, 0]
        v = [0, 0, 0, 0, 0, 0]
        m_hat = [0, 0, 0, 0, 0, 0]
        v_hat = [0, 0, 0, 0, 0, 0]
        # Update array for weights (final adam equation)
        updateGrads = [0, 0, 0, 0, 0, 0]
        # Define array of gradients for ease of calculation
        Gradients = [0, 0, 0, 0, 0, 0]

        # Array of Error Values
        errGraph = np.zeros((max_iterations,1))

        for iteration in range(max_iterations):
            out = self.forward_pass(self.activation)
            error = 0.5 * np.power((out - self.y), 2)
            errGraph[iteration] = np.sum(error)
            # TODO: I have coded the sigmoid activation, you have to do the rest
            self.backward_pass(out, self.activation)

            # Gradients for Output to HL2
            # update_weight_output
            Gradients[0] = np.dot(self.X_hidden2.T, self.deltaOut)
            # update_weight_output_b
            Gradients[1] = np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)

            # Gradients for HL2 to HL1
            # update_weight_hidden2
            Gradients[2] = np.dot(self.X_hidden1.T, self.deltaHidden2)
            # update_weight_hidden2_b
            Gradients[3] = np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden2)

            # Gradients for HL1 to Input
            # update_weight_hidden
            Gradients[4] = np.dot(self.X.T, self.deltaHidden1)
            # update_weight_hidden_b
            Gradients[5] = np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden1)

            # Calculate Adam Optimizer Values
            for i in range(0,6):
                # Calculate m & v estimates for output to HL2 weights
                m[i] = (beta1 * m[i]) + ((1 - beta1) * Gradients[i])
                v[i] = (beta2 * v[i]) + ((1 - beta2) * (Gradients[i] ** 2))
                # Calculate bias-corrected m_hat & v_hat for output to HL2 weights
                m_hat[i] = m[i]/(1 - beta1)
                v_hat[i] = v[i]/(1 - beta2)
                # Calculate final update value
                updateGrads[i] = (((alpha)/(np.sqrt(v_hat[i]) + epsilon)) * m_hat[i])

            self.W_output += updateGrads[0]
            self.Wb_output += updateGrads[1]
            self.W_hidden2 += updateGrads[2]
            self.Wb_hidden2 += updateGrads[3]
            self.W_hidden1 += updateGrads[4]
            self.Wb_hidden1 += updateGrads[5]

        print("Activation Function: ", self.activation)
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden1))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden2))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden1))
        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden2))
        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output) + "\n")
        return errGraph

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in_hidden1 = np.dot(self.X, self.W_hidden1) + self.Wb_hidden1
        # Hidden Layer Forward
        if activation == "sigmoid":
            self.X_hidden1 = self.__sigmoid(in_hidden1)
        if activation == "relu":
            self.X_hidden1 = self.__relu(in_hidden1)
        if activation == "tanh":
            self.X_hidden1 = self.__tanh(in_hidden1)

        # Hidden Layer 2 Out
        in_hidden2 = np.dot(self.X_hidden1, self.W_hidden2) + self.Wb_hidden2
        if activation == "sigmoid":
            self.X_hidden2 = self.__sigmoid(in_hidden2)
        if activation == "relu":
            self.X_hidden2 = self.__relu(in_hidden2)
        if activation == "tanh":
            self.X_hidden2 = self.__tanh(in_hidden2)

        # Output Layer Out dot product with Output Weights
        in_output = np.dot(self.X_hidden2, self.W_output) + self.Wb_output
        # Output Layer Forward
        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        if activation == "relu":
            out = self.__relu(in_output)
        if activation == "tanh":
            out = self.__tanh(in_output)
        return out

    # Added Function to Propogate Test Data Set through Trained Model
    def forward_test(self, activation, t):
        if t == "train":
            # pass our inputs through our neural network
            in_hidden1 = np.dot(self.X, self.W_hidden1) + self.Wb_hidden1
        if t == "test":
            # pass our inputs through our neural network
            in_hidden1 = np.dot(self.Xtest, self.W_hidden1) + self.Wb_hidden1
        # Hidden Layer Forward
        if activation == "sigmoid":
            self.X_test_hidden1 = self.__sigmoid(in_hidden1)
        if activation == "relu":
            self.X_test_hidden1 = self.__relu(in_hidden1)
        if activation == "tanh":
            self.X_test_hidden1 = self.__tanh(in_hidden1)

        # Hidden Layer 2 Out
        in_hidden2 = np.dot(self.X_test_hidden1, self.W_hidden2) + self.Wb_hidden2
        if activation == "sigmoid":
            self.X_hidden2 = self.__sigmoid(in_hidden2)
        if activation == "relu":
            self.X_hidden2 = self.__relu(in_hidden2)
        if activation == "tanh":
            self.X_hidden2 = self.__tanh(in_hidden2)

        # Output Layer Out dot product with Output Weights
        in_output = np.dot(self.X_hidden2, self.W_output) + self.Wb_output
        # Output Layer Forward
        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        if activation == "relu":
            out = self.__relu(in_output)
        if activation == "tanh":
            out = self.__tanh(in_output)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden2_delta(activation)
        self.compute_hidden1_delta(activation)

    # TODO: Implement other activation functions

    '''
        deltaOut is the key thing to change to edit the error derivative
    '''
    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        self.deltaOut = delta_output

    def compute_hidden2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden2))
        if activation == "relu":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__relu_derivative(self.X_hidden2))
        if activation == "tanh":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__tanh_derivative(self.X_hidden2))

        self.deltaHidden2 = delta_hidden_layer

    def compute_hidden1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer = (self.deltaHidden2.dot(self.W_hidden2.T)) * (self.__sigmoid_derivative(self.X_hidden1))
        if activation == "relu":
            delta_hidden_layer = (self.deltaHidden2.dot(self.W_hidden2.T)) * (self.__relu_derivative(self.X_hidden1))
        if activation == "tanh":
            delta_hidden_layer = (self.deltaHidden2.dot(self.W_hidden2.T)) * (self.__tanh_derivative(self.X_hidden1))

        self.deltaHidden1 = delta_hidden_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    # Get accuracy of Train and Test dataset
    def predict(self, activation, header = True):
        outputs = self.forward_test(activation, "train")
        print("Train Accuracy Results for ", activation, " activation function:")
        correct = 0
        if len(outputs) == len(self.y):
            for i in range(len(outputs)):
                # Both Outputs have to match
                # So for instance 1 output (1,0) must equal y (1,0)
                if np.around(outputs[i,0]) == self.y[i,0]:
                    if np.around(outputs[i, 1]) == self.y[i, 1]:
                        if np.around(outputs[i,2]) == self.y[i,2]:
                            #print("out:", np.around(outputs[i,0]), np.around(outputs[i,1]), np.around(outputs[i,2]), "actual:",self.y[i,0], self.y[i,1], self.y[i,2])
                            correct += 1
            print("Percent Correct: ", (correct/len(outputs))*100, "%")
            print("Mean Squared Error: ", np.around((np.sum(0.5 * np.power((outputs - self.y), 2))), decimals=8))
        outputs = self.forward_test(activation, "test")
        print("Test Accuracy Results for ", activation, " activation function:")
        correct = 0
        if len(outputs) == len(self.ytest):
            for i in range(len(outputs)):
                # Both Outputs have to match
                # So for instance 1 output (1,0) must equal y (1,0)
                if np.around(outputs[i, 0]) == self.ytest[i, 0]:
                    if np.around(outputs[i, 1]) == self.ytest[i, 1]:
                        if np.around(outputs[i, 2]) == self.ytest[i, 2]:
                            #print("out:", np.around(outputs[i,0]), np.around(outputs[i,1]), np.around(outputs[i,2]), "actual:",self.ytest[i,0], self.ytest[i,1], self.ytest[i,2])
                            correct += 1
            print("Percent Correct: ", (correct/len(outputs))*100, "%")
            print("Mean Squared Error: ", np.around((np.sum(0.5 * np.power((outputs - self.ytest), 2))), decimals=8), "\n")


if __name__ == "__main__":
    # Initialize Variables
    # Randomly Generate State for Train/Test Split
    s1 = 3
    seed(s1)
    state = randint(0,1000)
    s2 = 8
    seed(s2)
    randseed = randint(0,1000)
    max_iterations = 8000
    LR = .001
    testsize = .1
    h1 = 6
    h2 = 4

    # Train Sigmoid Model
    neural_network_sigmoid = NeuralNet("train.csv", "sigmoid", state, randseed, h1, h2, testsize)
    err_sigmoid = neural_network_sigmoid.train(max_iterations, LR)
    # Train ReLu Model
    neural_network_relu = NeuralNet("train.csv", "relu", state, randseed, h1, h2, testsize)
    err_relu = neural_network_relu.train(max_iterations, LR)
    # Train Tanh Model
    neural_network_tanh = NeuralNet("train.csv", "tanh", state, randseed, h1, h2, testsize)
    err_tanh = neural_network_tanh.train(max_iterations, LR)

    print("err s:\n", err_sigmoid)
    print("err r:\n", err_relu)
    print("err t:\n", err_tanh)

    print("Iterations|   LR|Test Size|Seed1|Seed2|state|Randseed|h1|h2|")
    print("      {itr}|{LR}|      {ts}|   {seed1}|{seed2}| {state}| {rs}| {h1}| {h2}|".format(itr = max_iterations, LR = LR, ts = testsize, seed1=s1, seed2=s2,rs = randseed,state=state,h1=h1,h2=h2))
    print()
    # Print Out Test Error for Each Model
    neural_network_sigmoid.predict("sigmoid")
    neural_network_relu.predict("relu")
    neural_network_tanh.predict("tanh")
