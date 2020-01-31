import numpy as np
from NeuralNet import *

X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

Ytarget = np.array([[0], [1], [1], [0]])

#X = np.array([[0, 0],
#              [1, 0],
#              [0, 1],
#              [1, 1]])
#
#Ytarget = np.array([[1, 0],
#                    [0, 1],
#                    [0, 1],
#                    [1, 0]])

def print_xor_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\n  X   => y_target =>   y_pred  => round(y_pred)')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} =>   {y}    => {z.round(5)} =>  {z.round()}')
        
def train_xor(net: Optimizer, inputs: Tensor, targets: Tensor, epochs: int = 10000):
    #trains the Neural Network using the given input and target
    train(net, inputs, targets, num_epochs=epochs)
    #Uses this trained Neural Network to predict the output with the same given input
    predictions = net.forward(inputs)
    #prints the results to compare with the expected output
    print_xor_results(inputs, targets, predictions)


###LINEAR LAYER --> Can't work
#net1 = NeuralNet([Linear(input_size=2, output_size=1),])
#train_xor(net1, X, Ytarget)_

##NON-LINEAR NETWORK (adding a non-linear activation layer: Tanh(), Sigmoid(), ...)

#Takes the dimensions of the inputs and outputs to size the NN
L_in=X.shape[1]
L_out=Ytarget.shape[1]

#Creation of the Neural Network
net2 = NeuralNet([
        # Add the layers here
        Linear(input_size=L_in, output_size=L_in),
        Tanh(),
        #ReLu(),
        #Sigmoid(),
        Linear(input_size=L_in, output_size=L_out)
])

#Use of this NN to construct the XOR function
train_xor(net2, X, Ytarget)
