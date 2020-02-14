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
cCT0=np.array(data['cleoConeThrust0'],dtype=float)
cCT1=np.array(data['cleoConeThrust1'],dtype=float)

X=(np.array([ norm(fWR2), norm(cCT0), norm(cCT1) ])).T
Ytarget=(np.array([np.array(data['isBB'],dtype=float)])).T

#Other data to test the trained NN
databis=csv2dict("../Data/data100_2.csv")

sph_bis=np.array(databis['sphericity'],dtype=float)
hMT2_bis=np.array(databis['harmonicMomentThrust2'],dtype=float)
hMT3_bis=np.array(databis['harmonicMomentThrust3'],dtype=float)
hMT4_bis=np.array(databis['harmonicMomentThrust4'],dtype=float)
fWR2_bis=np.array(databis['foxWolframR2'],dtype=float)
fWR3_bis=np.array(databis['foxWolframR3'],dtype=float)
cCT0_bis=np.array(databis['cleoConeThrust0'],dtype=float)
cCT1_bis=np.array(databis['cleoConeThrust1'],dtype=float)

X_bis=(np.array([ norm(fWR2_bis), norm(cCT0_bis), norm(cCT1_bis) ])).T
Ytarget_bis=(np.array([np.array(databis['isBB'],dtype=float)])).T


###FUNCTIONS_____________________________________________________________________________________________

def print_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\n  X   => y_target =>       y_pred      => round(y_pred)')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} =>   {y}    => {z.round(5)} =>  {z.round()}') #{float(z):.5f} =>    {z.round()}')
        
def train_NN(net: Optimizer, inputs: Tensor, targets: Tensor, loss_func: Loss = MeanSquareError(), epochs: int = 10000):
    train(net, inputs, targets, num_epochs=epochs, loss=loss_func)
    predictions = net.forward(inputs)
    print_results(inputs, targets, predictions)
    
    Loss_value=loss_func.loss(predictions,targets)
    print("\nloss function:", Loss_value)

def test_NN(net: Optimizer, inputs: Tensor, targets: Tensor, loss_func: Loss = MeanSquareError()):
    predictions = net.forward(inputs)
    print_results(inputs, targets, predictions)
    
    Loss_value=loss_func.loss(predictions,targets)
    print("\nloss function:", Loss_value)

###MAIN PROGRAM_________________________________________________________________________________________

##NON-LINEAR NETWORK (adding a non-linear activation layer: Tanh(), Sigmoid(), ...)
L_in=X.shape[1]
L_out=Ytarget.shape[1]

net = NeuralNet([
        # Add the layers here
        Linear(input_size=L_in, output_size=25),
        #Tanh(),
        #Linear(input_size=L_in, output_size=L_in),
        #ReLu(),
        Sigmoid(),
        Linear(input_size=25, output_size=L_out)
])

train_NN(net, X, Ytarget)
#print("\n\n\n")
#test_NN(net, X_bis, Ytarget_bis)
