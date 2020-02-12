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

class MeanSquareError(Loss): #inherits from 'Loss' class 
    
    def loss(self, predicted, actual):
        MSE=sum((predicted-actual)**2)
        return MSE
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        dMSE=2*(predicted-actual) #Derivative with respect to each variable of the 'predicted' tensor
        return dMSE
    
#-----------WIP----------------------------------------------------------------
class BinaryCrossEntropy(Loss): #inherits from 'Loss' class 
    
    def loss(self, predicted, actual):
        m = predicted.shape[0]
        cost = -(1/m) * (np.dot(actual.T, np.log(abs(predicted))) + np.dot((1 - actual).T, np.log(abs(1 - predicted))))
        return np.squeeze(cost)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        dcost = - (np.divide(actual,predicted) - np.divide(1 - actual, 1 - predicted)) #Derivative with respect to each variable of the 'predicted' tensor
        return dcost

#-------------------------------------------------------------------------------

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
def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          loss: Loss = MeanSquareError(),
          optimizer: Optimizer = SGD(),
          iterator: DataIterator = BatchIterator(),
          num_epochs: int = 5000) -> None:
    
    for epoch in range(num_epochs+1):
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
            
        # Print status every 50 iterations
        if epoch % 50 == 0:
            print(epoch, epoch_loss)
#    print("      ", epoch, "           ",epoch_loss)

