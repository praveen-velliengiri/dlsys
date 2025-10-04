"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist_impl(filename):
  with gzip.open(filename, 'rb') as f:
    magic, = struct.unpack('>I', f.read(4))
    dtype  = (magic >> 8) & 0xFF
    ndims  = magic & 0xFF
    
    #print("magic is :  ", hex(magic))
    #print("number of dimensions : ", ndims)

    shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndims))
    print(shape)

    dtypes = {0x08: np.uint8, 0x09: np.int8, 0x0B : np.int16,
              0x0C: np.int32, 0x0D: np.float32, 0x0E: np.float64}
    
    data = np.frombuffer(f.read(), dtype=dtypes[dtype])
    return data, shape
  
def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    #input data.
    rawdata, shape = parse_mnist_impl(image_filename)
    newshape = (shape[0], shape[1] * shape[2])
    X = rawdata.astype(np.float32).reshape(newshape)
    X = X / 255.0 

    #labels data.
    rawdata, shape = parse_mnist_impl(label_filename)
    Y = rawdata.astype(np.uint8)
    return (X, Y)

def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    #m x 1
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes= 1))
    #pick the logit associated with the true label
    true_logit = ndl.summation(ndl.multiply(Z, y_one_hot), axes=1)
    #loss_for_entire_batch #m x 1
    loss = ndl.add(log_sum_exp, ndl.negate(true_logit))

    return ndl.divide_scalar(ndl.summation(loss, axes = 0), Z.shape[0])

    '''
    #Z shape - m x k
    #y_one_hot - m x k
    batch_size = Z.shape[0] 

    #numerical valid.
    Z_max = np.max(Z, axis=1, keepdims=True)

    #log-sum-exp
    
    Z_stable  = Z - Z_max
    Z_correct = Z_stable - Z_stable[np.arange(batch_size), y_one_hot][:, np.newaxis]
    value = np.mean(np.log(np.sum(np.exp(Z_correct), axis=1)))

    #log-sum-exp (needle)
    Z_stable  = ndl.add(Z, ndl.negate(Z_max))
    z_neg     = ndl.negate(Z_stable[np.arange(batch_size), y_one_hot][:, np.newaxis])
    Z_correct = ndl.add(Z_stable, z_neg)
    sumof = ndl.Summation(ndl.log(ndl.Summation(ndl.exp(Z_correct), axis=1)),0)

    return ndl.divide_scalar(sumof, batch_size)
    '''


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    batch_size = batch
    num_class  = W2.numpy().shape[1]
    for i in range(0, X.shape[0], batch_size):
      x_numpy = X[i : i+batch_size]
      y_numpy = y[i : i+batch_size]

      x_ndl = ndl.Tensor.make_const(x_numpy)
      
      one_hot = np.zeros((batch_size, num_class), dtype=int)
      one_hot[np.arange(batch_size), y_numpy] = 1
      y_ndl = ndl.Tensor.make_const(one_hot)

      loss = softmax_loss(ndl.matmul(ndl.relu(ndl.matmul(x_ndl, W1)), W2), y_one_hot=y_ndl)
      loss.backward()
      W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
      W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
  
    return (W1, W2) 




    '''
    for i in range(0, X.shape[0], batch_size):
      x = X[i : i+batch_size] #batch_size x input_dim
      labels = y[i : i+batch_size] #batch_size

      Z_1 = np.maximum(x @ W1, 0) #batch_size x hidden_dim
      
      #batch_size x hidden_dim @ hidden_dim x num_classes

      e1 = np.exp(Z_1 @ W2)
      esum = np.sum(e1, axis=1, keepdims=True)
      softmax = e1/esum #batch_size x num_classes
      softmax[np.arange(x.shape[0]), labels] -= 1
      
      G_2 = softmax #batch_size x num_classes

      relu_mask = np.where(Z_1 > 0, 1, 0)
      G_1 = relu_mask * (G_2 @ W2.T) #batch_size x hidden_dim

      #input_dim x batch_size x batch_size x hidden_dim
      #input_dim x hidden_dim
      W1 -=  lr * ((x.T @ G_1)/batch_size)

      #hidden_dim x batch_size x batch_size x num_classes
      #hidden_dim x num_classes
      W2 -=  lr * ((Z_1.T @ G_2)/batch_size)
    '''


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
