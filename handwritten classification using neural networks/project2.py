import numpy as np

# #######################################################
# # Non linearity functions
# #######################################################


def relu(Z):
    """
    Implement the RELU function.
    Args:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    Internal parameters -- a python dictionary containing "Z" ; 
        stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    internal_params = Z
    return A, internal_params


def relu_backward(dA, internal_params):
    """
    Implement the backward propagation for a single RELU unit.
    Args:
    dA -- post-activation gradient, of any shape
    Internal params -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = internal_params
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ


def sigmoid(Z):
    """
    Implement the SIGMOID function.
    Args:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    Internal parameters -- a python dictionary containing "Z" ; 
        stored for computing the backward pass efficiently
    """

    A = 1 / (1 + np.exp(Z))

    internal_params = Z
    return A, internal_params


def sigmoid_backward(dA, internal_params):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Args:
    dA -- post-activation gradient, of any shape
    Internal params -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z= internal_params
    dZ=np.multiply(sigmoid(Z)*(1-sigmoid(Z)),dA)
    # raise NotImplementedError
    return dZ


def tanh(Z):
    """
    Implement the TANH function.
    Args:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    Internal parameters -- a python dictionary containing "Z" ; 
        stored for computing the backward pass efficiently
    """

    e_2_z = np.exp(2*Z)

    A = (e_2_z - 1) / (e_2_z + 1)

    internal_params = Z
    return A, internal_params


def tanh_backward(dA, internal_params):
    """
    Implement the backward propagation for a single TANH unit.
    Args:
    dA -- post-activation gradient, of any shape
    Internal params -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = internal_params
    Zt=tanh(Z)
    dzp=np.power(Zt,2)
    print(dzp.shape)
    dZ=np.multiply(dzp,dA)
    return dZ
    # raise NotImplementedError


# #######################################################
# # Start utility functions
# #######################################################

def initialize_parameters(X, Y, nb_units_per_hidden_layer):
    """
    Initialize the parameters for each layer. 
    Make sure that you have the right size. Refer to the neural network
    figure in the lab pdf.

    To initialize a random gaussian matrix use np.random.randn(d1, d2) * 0.05
    To initialize the zeros vector 

    Args:
        X -- the input matrix (get size of input)
        Y -- data labels
        nb_units_per_hidden_layer -- list of integer: nb_unit per hidden layer

    Returns:
        params -- python dictionary containing your parameters:
                        Wl -- weight matrix of shape (n_layer_l, n_layer_l-1)
                        bl -- bias vector of shape (n_layer_l, 1)
    """
    # Your code here
    np.random.seed(1)
    params = {}
    L = len(nb_units_per_hidden_layer)
    params['W' + str(1)] = np.random.randn(nb_units_per_hidden_layer[0],X.shape[0] ) * 0.05
    params['b' + str(1)] = np.zeros((nb_units_per_hidden_layer[0], 1))

    for i in range(1, L):
        params['W' + str(i+1)] = np.random.randn(nb_units_per_hidden_layer[i], nb_units_per_hidden_layer[i - 1]) * 0.01
        params['b' + str(i+1)] = np.zeros((nb_units_per_hidden_layer[i], 1))
    params['W' + str(L+1)]= np.random.randn(1, nb_units_per_hidden_layer[L-1]) * 0.05
    params['b' + str(L+1)]= np.zeros((1,1))
    return params
    # raise NotImplementedError


def linear_forward_calculation(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Args:
    A -- previous layer output
    W -- weights matrix
    b -- bias vector

    Returns:
    Z -- the input of the activation function 
    internal_params -- a python dictionary containing "A", "W" and "b", to use for backprop
    """
    # Your code here
    # print(W.shape, A.shape, b.shape)
    Z=np.dot(W,A)+b

    return Z
    # raise NotImplementedError


def linear_activation_calculation(A, W, b, activation_function):
    """
    Implement the activation for the forward propagation

    Args:
    A -- activations of the previous layer
    W -- weights matrix
    b -- bias vector
    activation -- "sigmoid" or "relu" or "tanh"

    Returns:
    A -- the output of the activation function for the current layer
    internal_params -- a python dictionary containing "linear_internal_params" from the linear forward pass
    and "activation_internal_params"
    """

    # Your code here
    return activation_function(linear_forward_calculation(A, W, b))
    # raise NotImplementedError


def L_Layers_forward_model(train_x, params):
    """
    Implement forward pass. The following steps will helps you:
    -   Calculate the linear forward pass using the corresponding params.
    -   Apply the activation function :
        - Relu activation for hidden layers
        - Sigmoid or Tanh for the last layer (remember there is only one unit the output layer)
    -   Save the internal params for each layer to use them in the backward pass

    Args:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of the function initialize_parameters()

    Returns:
    AL -- The last activation function.
    internal_params -- list of internal_params containing:
                every internal_params of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the internal_params of linear_sigmoid_forward() (there is one, indexed L-1)
    """


    A=train_x
    L= int(len(params)/2)
    internalParams=[]
    # print(params)
    for i in range(1,int(L)):
        A_p=A
        A,internal = linear_activation_calculation(A_p,params['W'+str(i)],params['b'+str(i)],relu)
        # print(A.shape)
        internalParams.append(internal)
    AL,internal=linear_activation_calculation(A,params['W'+str(L)],params['b'+str(L)],tanh)
    internalParams.append(internal)
    return AL,internalParams
    # raise NotImplementedError


def compute_error_cross_dataset(AL, train_y):
    """
    Implement the cost function using the function defined in the micro project pdf file.

    Args:
    AL -- Output of the function L_Layers_forward_model
    train_y -- The label of the data : Binary vector. train_y.shape = (1, nb_dataset_elements) 

    Returns:
    cost -- 
    """
    # print(train_y.shape)
    nb = train_y.shape[0]
    error=np.power(np.add(train_y,-AL),2)*1/nb
    return error
    # raise NotImplementedError


def linear_backward_calculation(dZ, internal_params):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output calculation (of current layer l)
    internal_params -- tuple of values:
    -   A_prev: activation of layer l-1 <=> the input for layer l
    -   W, b: params for layer l

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = internal_params
    nb = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW =np.multiply((np.dot(dZ, A_prev.T)),1/nb)
    db = np.multiply ((np.sum(dZ, axis=1, keepdims=True),1/nb))
    dA_prev = np.dot(W.T, dZ)
    # raise NotImplementedError
    return dA_prev,dW,db

def linear_activation_backward_calculation(dA, internal_params, activation_function):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l 
    internal_params -- tuple of values (the output of the function linear_activation_calculation):
    -   linear_internal_params
    -   activation_internal_params 


    activation -- the activation to be used in this layer, stored as a text string: "tanh" or "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    if activation_function == "relu":
        dZ = relu_backward(dA, internal_params)
    elif activation_function == "sigmoid":
        dZ = sigmoid_backward(dA, internal_params)
    else:
        dZ=tanh_backward(dA,internal_params)
    dA_prev, dW, db = linear_backward_calculation(dZ, internal_params)
    return dA_prev, dW, db

    # raise NotImplementedError


def L_Layers_backword_model(AL, train_y, internal_params):
    """
    Implement the backward propagation for the model using the functions above

    Arguments:
    AL -- output of the forward propagation (L_Layers_forward_model())
    Y -- The label of the data : Binary vector. train_y.shape = (1, nb_dataset_elements) 
    internal_params -- list of internal_params containing:
                every internal_params of linear_activation_forward() with "relu" (it's internal_params[l], for l in range(L-1) i.e l = 0...L-2)
                the internal_params of linear_activation_forward() with "sigmoid" (it's internal_params[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = int(len(internal_params)/2)
    # nb = AL.shape[1]
    print(AL.shape)
    train_y = train_y.reshape(AL.shape)
    dAL = compute_error_cross_dataset(AL, train_y)
    curr_params = internal_params[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward_calculation(dAL, curr_params,"tanh")
    for i in reversed(range(L - 1)):
        curr_params = internal_params[i]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward_calculation(grads["dA" + str(i + 1)], curr_params,"relu")
        grads["dA" + str(i + 1)] = dA_prev_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp
    return grads

    # raise NotImplementedError


def update_model_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing the parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing the new parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) /2  # number of layers in the neural network

    for l in range(int(L)):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
    # raise NotImplementedError


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID / TANH.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px)
    Y --The label of the data : Binary vector. train_y.shape = (1, nb_dataset_elements) 
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    # #######################################################################
    # # 1. Initialize your paramerters
    # #######################################################################
    params=initialize_parameters(X, Y, layers_dims)
    # #######################################################################
    # # 2. Number of iteration loop
    # #######################################################################
    for i in range(num_iterations):
    # ###################################################################
    # # 2.1. Forword pass calculation
    # ###################################################################
        AL,internal=L_Layers_forward_model(X,params)
    # ###################################################################
    # # 2.2. Compute Cost and save in variables to create a timeserie
    # ###################################################################
        cost=compute_error_cross_dataset(AL,Y)
    # ###################################################################
    # # 2.3. Backword pass calculation
    # ###################################################################
        grads=L_Layers_backword_model(np.transpose(AL),Y,internal)
    # ###################################################################
    # # 2.4. Update parameter
    # ###################################################################
        params=update_model_parameters(params,grads,learning_rate)
    # raise NotImplementedError
    return params

def predict(X, Y, params):
    """
    Implements a prediction function using L_Layers_forward_model

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px)
    Y --The label of the data : Binary vector. train_y.shape = (1, nb_dataset_elements) 
    params -- parameters learnt by the model.

    Returns:
    output_vector -- Binary vector that contains the prediction:
    -   1 correct prediction
    -   0 wrong prediction 
    PLEASE NOTE that the values (0, 1) are not the labels.
    accuracy -- overall accuracy #correct predictions / #predections
    """
    AL=L_Layers_forward_model(X,params)
    diff=np.sum(AL,-Y)
    out = np.array(diff, copy=True)
    out[diff != 0] = 0
    out[diff == 0] = 1
    return out
    # raise NotImplementedError
