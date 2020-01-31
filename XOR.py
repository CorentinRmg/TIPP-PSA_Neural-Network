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

#X = np.array([[-354, 342]])
#
#Ytarget = np.array([[42]])

def print_xor_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\n  X   => y_target =>   y_pred  => round(y_pred)')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} =>   {y}    => {z.round(5)} =>  {z.round()}') #{float(z):.5f} =>    {z.round()}')
        
def train_xor(net: Optimizer, inputs: Tensor, targets: Tensor, epochs: int = 10000):
    train(net, inputs, targets, num_epochs=epochs)
    predictions = net.forward(inputs)
    print_xor_results(inputs, targets, predictions)


###LINEAR LAYER --> Can't work
#net1 = NeuralNet([Linear(input_size=2, output_size=1),])
#train_xor(net1, X, Ytarget)_

##NON-LINEAR NETWORK (adding a non-linear activation layer: Tanh(), Sigmoid(), ...)
net2 = NeuralNet([
        # Add the layers here
        Linear(input_size=2, output_size=2),
        #Tanh(),
        #ReLu(),
        Sigmoid(),
        Linear(input_size=2, output_size=1)
])

train_xor(net2, X, Ytarget)
