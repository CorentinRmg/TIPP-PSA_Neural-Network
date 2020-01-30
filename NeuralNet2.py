...
   Code inspired by https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
...

import numpy as np

nn_architecture = [
    {"input_dim": 4, "output_dim": 4, "activation": "tanh"},      #definition of the architecture used (layers)
    {"input_dim": 4, "output_dim": 6, "activation": "tanh"},
    {"input_dim": 6, "output_dim": 6, "activation": "tanh"},
    {"input_dim": 6, "output_dim": 4, "activation": "tanh"},
    {"input_dim": 4, "output_dim": 2, "activation": "sigmoid"}
]

def sigmoid(Z):                            #Definition of the activations functions
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def tanh(Z):
    return np.tanh(Z)
    
def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;
    
def tanh_backward(dA, Z):
    return dA * (1 - np.dot(tanh(Z),tanh(Z)))

def init_parameters_random(nn_architecture, seed=2):           #Initialisation of random W and b
    np.random.seed(seed)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        #Fill 'W' and 'b' values with random values
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1 
    return params_values

def init_layers(nn_architecture, parameters = 0, seed = 2):        #Initialisation of the layers and the seed
    #Either we choose to use random values, either we give values already known
    if parameters == 0:
        params_values=init_parameters_random(nn_architecture, seed)
    else:
        params_values = parameters
    return params_values



                                                                  #Computing of Z, and then of its activated function
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation = "relu"):     
    #Computing the value using the parameters
    Z_curr = np.dot(W_curr, A_prev) + np.squeeze(b_curr)
    
    #Selection of the activation function
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    elif activation is "tanh":
        activation_func = tanh
    else:
        raise Exception('Non-supported activation function')
        
    #Returning calculated A and Z    
    return activation_func(Z_curr), Z_curr
    
def full_forward_propagation(X, params, nn_architecture):           #forward propagation for all the layers
    #The first value for A is X, the data given
    A_curr = X
    #Creation of a temporary memory
    memory = {}    
    
    for idx, layer in enumerate(nn_architecture):
        #The layer begin at 1
        layer_idx = idx + 1
        #Previous A is used in the computing
        A_prev = A_curr
        W_curr = params['W' + str(layer_idx)]
        b_curr = params['b' + str(layer_idx)]
        #Computing of A and Z
        (A_curr, Z_curr) = single_layer_forward_propagation(A_prev, W_curr, b_curr, layer["activation"])
        #Saving the values
        memory['W'+str(layer_idx)] = W_curr
        memory['b'+str(layer_idx)] = b_curr
        memory['A'+str(idx)] = A_curr
        memory['Z'+str(layer_idx)] = Z_curr
    #Returning A_curr = Y_hat and the memory, to update the values    
    return A_curr,memory

def Loss_function(Y_hat, Y):                                      #Loss function definition
    #number of examples
    m = Y_hat.shape[0]
    #Necessary to get the good dimensions
    Y=np.squeeze(np.asarray(Y))
    #Calculating the loss function with crossentropy
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)
        
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):      #Backward computing for one layer
    m = A_prev.shape[0]
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    elif activation is "tanh":
        backward_activation_func = tanh_backward
    else:
        raise Exception('Non-supported activation function')
    
    #Computing the derivative vlaues
    dZ_curr = backward_activation_func(dA_curr, Z_curr)

    dW_curr = np.dot(dZ_curr, A_prev.T) / m

    db_curr = np.sum(dZ_curr, axis=0, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):        #Backward computing for all the layers
    #Memory to stock the values
    grads_values = {}
    m = Y_hat.shape[0]
    Y = Y.reshape(Y_hat.shape)
   
   #Initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    return grads_values
    
def update(params_values, grads_values, nn_architecture, learning_rate):              #Algorithm for the updating of 'W' and 'b' values
    for layer_idx, layer in enumerate(nn_architecture, 1):
        #Substract 'dW' and 'db' from 'W' and 'b'
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values
    
def train(X, Y, nn_architecture, epochs, learning_rate, verbose = False, callback = None, parameters=0):
    params_values = init_layers(nn_architecture, parameters)
    cost_history = []
    
    for i in range(epochs):
        #Going forward
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)      
        #Computing of the loss function
        cost = Loss_function(Y_hat, Y)
        cost_history.append(cost)
  
        #Computing of the gradient values
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        #Substraction of the grads values
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        #Print the values only every 50 iterations
        if(i % 50 == 0):
            if(verbose):
                print(i, cost)
            if(callback is not None):
                callback(i, params_values)
    return Y_hat
