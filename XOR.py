import numpy as np
from cookingNeuralNet import *

X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

Ytarget = np.array([[0], [1], [1], [0]])

def print_xor_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\n  X   => y_target => y_pred  => round(y_pred)')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} =>   {y}    => {float(z):.5f} =>    {z.round()}')
        
def train_xor(net: Optimizer, inputs: Tensor, targets: Tensor, epochs: int = 2000):
    train(net, inputs, targets, num_epochs=epochs)
    predictions = net.forward(inputs)
    print_xor_results(inputs, targets, predictions)


###LINEAR LAYER --> Can't work
#net1 = NeuralNet([Linear(input_size=2, output_size=1),])
#train_xor(net1, X, Ytarget)_


##NON-LINEAR NETWORK (adding an )
net2 = NeuralNet([
        # Add the layers here
        Linear(input_size=2, output_size=2),
        Tanh(),
        Linear(input_size=2, output_size=1)
])

train_xor(net2, X, Ytarget)