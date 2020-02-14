import numpy as np
from NeuralNet_overfittingcurve import *
from DataReader import *
import matplotlib.pyplot as plt

###DATA__________________________________________________________________________________________________

#Data for training
data=csv2dict("../Data/data5000.csv")

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
databis=csv2dict("../Data/data5000_2.csv")

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


###MAIN PROGRAM_________________________________________________________________________________________

##NON-LINEAR NETWORK (adding a non-linear activation layer: Tanh(), Sigmoid(), ...)
L_in=X.shape[1]
L_out=Ytarget.shape[1]
    
NN = NeuralNet([
        # Add the layers here
        Linear(input_size=L_in, output_size=25),
        Sigmoid(),
        Linear(input_size=25, output_size=L_out)
])

##Get the data
[ number_of_iteration, training_curve, testing_curve ] = train_and_testOverFitting(NN,X,X_bis,Ytarget,Ytarget_bis, num_epochs=500)

##Display results
plt.figure()
plt.plot(number_of_iteration,training_curve)
plt.plot(number_of_iteration,testing_curve)
#plt.yscale('log')
plt.xlabel("#iterations")
plt.ylabel("loss function value")
plt.show()