'''
Programm to test a specific trained Neural Network named "net" (the exact name is important!)
with some given data.
'''

import numpy as np
from NeuralNet import *
from DataReader import *

###DATA__________________________________________________________________________________________________

#Data for testing
data=csv2dict("../Data/data100.csv")

sph=np.array(data['sphericity'],dtype=float)
hMT2=np.array(data['harmonicMomentThrust2'],dtype=float)
hMT3=np.array(data['harmonicMomentThrust3'],dtype=float)
hMT4=np.array(data['harmonicMomentThrust4'],dtype=float)
fWR2=np.array(data['foxWolframR2'],dtype=float)
fWR3=np.array(data['foxWolframR3'],dtype=float)

X=(np.array([ norm(fWR2), norm(hMT2), norm(hMT4) ])).T
Ytarget=(np.array([np.array(data['isBB'],dtype=float)])).T


###FUNCTIONS_____________________________________________________________________________________________

def print_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\n  X   => y_target =>       y_pred      => round(y_pred)')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} =>   {y}    => {z.round(5)} =>  {z.round()}') #{float(z):.5f} =>    {z.round()}')

def test_NN(net: Optimizer, inputs: Tensor, targets: Tensor, loss_func: Loss = MeanSquareError()):
    predictions = net.forward(inputs)
    print_results(inputs, targets, predictions)    
    Loss_value=loss_func.loss(predictions,targets)
    print("\nloss function:", Loss_value)

###MAIN PROGRAM_________________________________________________________________________________________

##NON-LINEAR NETWORK (adding a non-linear activation layer: Tanh(), Sigmoid(), ...)
L_in=X.shape[1]
L_out=Ytarget.shape[1]

test_NN(net, X, Ytarget)