import numpy as np
from NeuralNet import *
from DataReader import *
from Sidefunc import *
import pickle

###DATA__________________________________________________________________________________________________

Dataset='data500'

#Data for training
data_training_path='../Data/'+Dataset+'.csv'
data=csv2dict(data_training_path)
sph=np.array(data['sphericity'],dtype=float)
hMT2=np.array(data['harmonicMomentThrust2'],dtype=float)
hMT3=np.array(data['harmonicMomentThrust3'],dtype=float)
hMT4=np.array(data['harmonicMomentThrust4'],dtype=float)
fWR2=np.array(data['foxWolframR2'],dtype=float)
fWR3=np.array(data['foxWolframR3'],dtype=float)
cCT0=np.array(data['cleoConeThrust0'],dtype=float)
cCT1=np.array(data['cleoConeThrust1'],dtype=float)
cCT2=np.array(data['cleoConeThrust2'],dtype=float)
cCT3=np.array(data['cleoConeThrust3'],dtype=float)

X=(np.array([ norm(fWR2), norm(cCT0), norm(cCT3) ])).T
Ytarget=(np.array([np.array(data['isBB'],dtype=float)])).T

#Other data to test the trained NN
data_testing_path='../Data/'+Dataset+'_2.csv'
databis=csv2dict(data_testing_path)

sph_bis=np.array(databis['sphericity'],dtype=float)
hMT2_bis=np.array(databis['harmonicMomentThrust2'],dtype=float)
hMT3_bis=np.array(databis['harmonicMomentThrust3'],dtype=float)
hMT4_bis=np.array(databis['harmonicMomentThrust4'],dtype=float)
fWR2_bis=np.array(databis['foxWolframR2'],dtype=float)
fWR3_bis=np.array(databis['foxWolframR3'],dtype=float)
cCT0_bis=np.array(databis['cleoConeThrust0'],dtype=float)
cCT1_bis=np.array(databis['cleoConeThrust1'],dtype=float)
cCT2_bis=np.array(databis['cleoConeThrust2'],dtype=float)
cCT3_bis=np.array(databis['cleoConeThrust3'],dtype=float)

X_bis=(np.array([ norm(fWR2_bis), norm(cCT0_bis), norm(cCT3_bis) ])).T
Ytarget_bis=(np.array([np.array(databis['isBB'],dtype=float)])).T


###FUNCTIONS_____________________________________________________________________________________________

def print_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\n  X   => y_target =>       y_pred      => round(y_pred)')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} =>   {y}    => {z.round(5)} =>  {z.round()}') #{float(z):.5f} =>    {z.round()}')

def test_NN_ROC(net: Optimizer, inputs: Tensor, targets: Tensor, loss_func: Loss = MeanSquareError()):
    predictions = net.forward(inputs)
    print_results(inputs, targets, predictions)
    Loss_value=loss_func.loss(predictions,targets)
    return predictions
    
def make_ROC(var, isBB, nbins = 500, xrange=None, lower_is_better=False):

    if xrange is None:
        xrange = [var.min(), var.max()]
    bins = np.linspace(xrange[0], xrange[1], nbins)
    bb, nobb = sort(var,isBB)
    dist_sgn, _ = np.histogram(bb, bins=bins)
    dist_bkg, _ = np.histogram(nobb, bins=bins)
    dist_sgn = dist_sgn/dist_sgn.sum()
    dist_bkg = dist_bkg/dist_bkg.sum()
    if not lower_is_better:
        dist_sgn = np.array(tuple(reversed(dist_sgn)))
        dist_bkg = np.array(tuple(reversed(dist_bkg)))
    eff_sgn = [sum(dist_sgn[:i+1]) for i in range(len(dist_sgn))]
    eff_bkg = [sum(dist_bkg[:i+1]) for i in range(len(dist_bkg))]
    return eff_bkg, [1-i for i in eff_sgn]
    
it = '140'                                #Choose the parameters defined with it numbers of iterations
net = pickle.load(open("./NNbackups/TrainedNN_normalized_"+it+"-Iterations.pkl","rb"))            #can be modified deleting the "normalized"
N = test_NN(net, X_bis, Ytarget_bis)      #Either you test it, either you try it on the training, deleting the "_bis"

plt.figure(3)
plt.plot(*make_ROC(N,np.array(list(map(float,databis['isBB']))), lower_is_better=True), label=it+' iteration')        #Has to replace "databis" by "data" for the training
plt.xlabel('sgn efficiency', horizontalalignment='right', x=1)
plt.ylabel('bbbar rejection', horizontalalignment='right', y=1)
plt.legend(loc=3, fontsize='x-small')
plt.title('ROC curve for testing')
