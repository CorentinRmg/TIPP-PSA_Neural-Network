import numpy as np
from NeuralNet_overfittingcurve import *
from DataReader import *
import matplotlib.pyplot as plt

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

X_bis=(np.array([ norm(fWR2_bis), norm(cCT0_bis) , norm(cCT3_bis) ])).T
Ytarget_bis=(np.array([np.array(databis['isBB'],dtype=float)])).T


###MAIN PROGRAM_________________________________________________________________________________________

##NON-LINEAR NETWORK (adding a non-linear activation layer: Tanh(), Sigmoid(), ...)
L_in=X.shape[1]
L_out=Ytarget.shape[1]
middle_layer_size=10
    
NN = NeuralNet([
        # Add the layers here
        Linear(input_size=L_in, output_size=middle_layer_size),
        Sigmoid(),
        Linear(input_size=middle_layer_size, output_size=L_out),
])

##Get the data
num_iterations=150
[ number_of_iteration, training_curve, testing_curve, initial_bias ] = train_and_testOverFitting(NN,X,X_bis,Ytarget,Ytarget_bis, num_epochs=num_iterations)

biased_testing_curve=[]
for i in testing_curve:
    biased_testing_curve.append(i+initial_bias)

##Display results
plt.figure("Overfitting "+str(Dataset)+" "+str(middle_layer_size)+"neurons "+str(num_iterations)+"iterations")
plt.plot(number_of_iteration,training_curve,label="Training")
plt.plot(number_of_iteration,biased_testing_curve,label="Testing")
#plt.yscale('log')
plt.title('Overfitting curve for a middle layer of '+ str(middle_layer_size) + ' neurons\nDataset = ' + Dataset + ', learning rate = 0.01')
plt.legend()
plt.xlabel("#iterations")
plt.ylabel("loss function value")
#plt.gca().set_ylim(30,55)
plt.show()