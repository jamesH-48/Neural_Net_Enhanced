Assignment 2
James Hooper ~ NETID: jah171230
Hritik Panchasara ~ NETID: hhp160130
------------------------------------------------------------------------------------------------------------------------------------
- For this assignment we used PyCharm to create/edit/run the code. 
- In PyCharm the code should run by simply pressing the Run dropdown, then clicking run making sure you are running NeuralNetFinal
- The dataset is public on a Github account that one of us own. 
- In if __name__ == '__main__' , you can select the parameters you wish to choose. 
- s1: Seed for the randomized state for the train-test-split tool
- s2: See for the randomized seed (randseed) for the randomized initialized weights
- max_iterations: the number of iterations for the training
- h1: the number of nodes/neurons for the first hidden layer
- h2: the number of nodes/neurones for the second hidden layer
- REMINDER: the Adam Optimizer is implemented in the train function of the NeuralNet class.
- The parameters for the Adam Optimizer are initially the same for each model for each activation function. The parameters are explained in the report.
- The calls are split into three seperate sections
	- Initialize and Train Models for each activation function (Sigmoid, ReLu, Tanh)
	- Print Mean Squared Error values for each trained model (this hopefully on certain cases will be decreasing)
	- Print out the formatting of the parameters entered & print the predicted error/accuracy of the Train and Test datasets
------------------------------------------------------------------------------------------------------------------------------------
- The code is based off of the initial code given by Professor Nagar so the beginning comments should explain the rest of the code.
- The main thing to mention is that the ReLu & Tanh activation functions are based off the way the Sigmoid activation function was set and the Second Hidden Layer was based off the way the First Hidden Layer was set in accordance to the output layer.
- Preprocessing is done to alter the classification output. Originally the dataset gives the output as 0 or 1. This is changed into a 2 output system where (0,1) = 0 & (1,0) = 1.
- Link to Dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication
------------------------------------------------------------------------------------------------------------------------------------
Libraries Used:
import numpy as np
import pandas as pd
from random import seed, randint
from sklearn.model_selection import train_test_split
------------------------------------------------------------------------------------------------------------------------------------
Just in case. To import libraries/packages in PyCharm.
- Go to File.
- Press Settings.
- Press Project drop down.
- Press Project Interpreter.
- Press the plus sign on the top right box, should be to the right of where it says "Latest Version".
- Search and Install packages as needed.
- For this assignment the packages are: pandas, numpy, and scikit-learn.