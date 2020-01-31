import numpy as np
from NeuralNet import *
from DataReader import *

data=csv2dict("./Data/data100.csv")

X=(np.array([np.array(data['sphericity'],dtype=float)])).T
Ytarget=(np.array([np.array(data['isBB'],dtype=float)])).T

#X = np.array([[0, 0],
#              [1, 0],
#              [0, 1],
#              [1, 1]])
#
#Ytarget = np.array([[0], [1], [1], [0]])

def print_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\n  X   => y_target =>       y_pred      => round(y_pred)')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} =>   {y}    => {z.round(5)} =>  {z.round()}') #{float(z):.5f} =>    {z.round()}')
        
def train_NN(net: Optimizer, inputs: Tensor, targets: Tensor, epochs: int = 10000):
    train(net, inputs, targets, num_epochs=epochs)
    predictions = net.forward(inputs)
    print_results(inputs, targets, predictions)

##NON-LINEAR NETWORK (adding a non-linear activation layer: Tanh(), Sigmoid(), ...)
L_in=X.shape[1]
L_out=Ytarget.shape[1]

net2 = NeuralNet([
        # Add the layers here
        Linear(input_size=L_in, output_size=L_in),
        Tanh(),
        #ReLu(),
        #Sigmoid(),
        Linear(input_size=L_in, output_size=L_out)
])

train_NN(net2, X, Ytarget)