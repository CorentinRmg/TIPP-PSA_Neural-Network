'''
Code inspired by https://gitlab.in2p3.fr/ccin2p3-support/formations/workshops-gpu/04-2019/deep-learning
'''

import numpy as np
from typing import (Dict, Tuple, Callable, Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor
Func = Callable[[Tensor], Tensor] #Function which takes a tensor and return a tensor

np.random.seed(99)

##Loss Function 
class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class SumSquaredError(Loss): #inherits from 'Loss' class 
    
    def loss(self, predicted, actual):
        SE=sum((predicted-actual)**2)
        return SE
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        dMSE=2*(predicted-actual) #Derivative with respect to each variable of the 'predicted' tensor
        return dMSE

##Layers
class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError
        
class Linear(Layer):
    """
    Inputs are of size (batch_size, input_size)
    Outputs are of size (batch_size, output_size)
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # Inherit from base class Layer
        super().__init__() #call the __init__ function of the superclass (here: Layer)
        # Initialize the weights and bias with random values
        np.random.seed(99)
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (batch_size, input_size)
        """
        self.inputs = inputs
        # Compute here the feed forward pass
        Z=inputs @ self.params["w"] + self.params["b"]
        return Z
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        grad shape is (batch_size, output_size)
        """
        # Compute here the gradient parameters for the layer
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        # Compute here the feed backward pass
        dA = grad @ self.params["w"].T
        return dA
    
class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: Func, f_prime: Func) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


##Functions
def tanh(x: Tensor) -> Tensor:
    # Write here the tanh function
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    # Write here the derivative of the tanh
    return 1-np.tanh(x)**2

class Tanh(Activation): #creates an activation layer using tanh as activation function
    def __init__(self):
        super().__init__(tanh, tanh_prime)
        

def sigmoid(x: Tensor) -> Tensor:
    # Write here the sigmoid function
    return 1/(1+np.exp(-x))

def sigmoid_prime(x: Tensor) -> Tensor:
    # Write here the derivative of the sigmoid
    sig=sigmoid(x)
    return sig*(1-sig)

class Sigmoid(Activation):  #creates an activation layer using a sigmoïd as activation function
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)

def relu(x: Tensor) -> Tensor:
    return np.maximum(0,x)

def relu_backward(x: Tensor) -> Tensor:
    M=x>0 #gives a matrix with 'True'/'False' values
    return M.astype(int)

class ReLu(Activation):  #creates an activation layer using ReLu as activation function
    def __init__(self):
        super().__init__(relu, relu_backward)

##Neural Network
class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """
        The forward pass takes the layers in order
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        """
        The backward pass is the other way around
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad       #yield~return for generators

##Optimizer
class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

#Stochastic Gradient Descent
class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr #lr=learning rate
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads(): 
            # Write here the parameters update
            param-=self.lr * grad   #gradient descent formula


### DATA

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError
        
class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)



### Training    
def test_NN(net: Optimizer, inputs: Tensor, targets: Tensor, loss_func: Loss = SumSquaredError()) -> float:
    predictions = net.forward(inputs)
    
    Loss_value=loss_func.loss(predictions,targets)
    return float(Loss_value)

def train_and_testOverFitting(net: NeuralNet,
          inputs: Tensor,
          inputs_testing: Tensor,
          targets: Tensor,
          targets_testing: Tensor,
          loss: Loss = SumSquaredError(),
          optimizer: Optimizer = SGD(),
          iterator: DataIterator = BatchIterator(),
          num_epochs: int = 5000) -> list:
    '''
    trains the neural network and also returns a list of lists containing the calculated loss functions for some given number of iterations (every 50 iterations)
    with the loss function for this neural network working on other testing data
    '''
    
    loss_training=[]
    loss_testing=[]
    number_of_iteration=[]
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            # Write here the various steps (in order) needed 
            # at each epoch
            
            #feed forward
            predicted=net.forward(batch.inputs)
            
            #calculation of the loss function and of gradients by comparison between 'predicted' and 'target'
            epoch_loss += loss.loss(predicted,batch.targets)
            grad=loss.grad(predicted,batch.targets)
            
            #feed backward
            net.backward(grad)
            
            #Update of NeuralNetwork with new w and b with respect to grad
            optimizer.step(net)
            
         
        #Print status every 50 iterations and add infos to the lists to return
        if ((epoch % 50 == 0)  or epoch in range(1,150)): #and (epoch != 0))
            print(epoch, "iterations") #, epoch_loss)
            
            number_of_iteration.append(epoch)
            
            loss_train_i=test_NN(net, inputs, targets)
            loss_training.append(loss_train_i)
            print("training loss =", loss_train_i)
            
            loss_test_i=test_NN(net, inputs_testing, targets_testing)
            loss_testing.append(loss_test_i)
            print("testing loss  =", loss_test_i)
            
            #Note the difference between first training loss and first testing loss, in order to align the curves
            if epoch == 0:
                diff_train_test = loss_train_i-loss_test_i
        

            
    return [number_of_iteration, loss_training, loss_testing, diff_train_test]
