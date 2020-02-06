import numpy as np
from NeuralNet import *
from DataReader import *

###DATA__________________________________________________________________________________________________

#Data for training
data=csv2dict("../Data/data100.csv")

sph=np.array(data['sphericity'],dtype=float)
hMT2=np.array(data['harmonicMomentThrust2'],dtype=float)
hMT3=np.array(data['harmonicMomentThrust3'],dtype=float)
hMT4=np.array(data['harmonicMomentThrust4'],dtype=float)
fWR2=np.array(data['foxWolframR2'],dtype=float)
fWR3=np.array(data['foxWolframR3'],dtype=float)

X=(np.array([hMT2, hMT4, fWR2, fWR3 ])).T
Ytarget=(np.array([np.array(data['isBB'],dtype=float)])).T

#Other data to test the trained NN
data2=csv2dict("../Data/data200.csv")

sph_bis=np.array(data2['sphericity'],dtype=float)
hMT2_bis=np.array(data2['harmonicMomentThrust2'],dtype=float)
hMT3_bis=np.array(data2['harmonicMomentThrust3'],dtype=float)
hMT4_bis=np.array(data2['harmonicMomentThrust4'],dtype=float)
fWR2_bis=np.array(data2['foxWolframR2'],dtype=float)
fWR3_bis=np.array(data2['foxWolframR3'],dtype=float)

X2=(np.array([hMT2_bis, hMT4_bis, fWR2_bis, fWR3_bis ])).T
Ytarget2=(np.array([np.array(data2['isBB'],dtype=float)])).T


###FUNCTIONS_____________________________________________________________________________________________

def print_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\n  X   => y_target =>       y_pred      => round(y_pred)')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} =>   {y}    => {z.round(5)} =>  {z.round()}') #{float(z):.5f} =>    {z.round()}')
        
def train_NN(net: Optimizer, inputs: Tensor, targets: Tensor, epochs: int = 10000):
    train(net, inputs, targets, num_epochs=epochs)
    predictions = net.forward(inputs)
    print_results(inputs, targets, predictions)

def test_NN(net: Optimizer, inputs: Tensor, targets: Tensor, loss_func: Loss = MeanSquareError()):
    predictions = net.forward(inputs)
    print_results(inputs, targets, predictions)    
    Loss_value=loss_func.loss(predictions,targets)
    print("\nFonction de coût:", Loss_value)

###MAIN PROGRAM_________________________________________________________________________________________

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
print("\n\n\n")
test_NN(net2, X2, Ytarget2)