import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    
    C,H,W = input_dim
    HH = filter_size
    WW = filter_size
    stride = 1
#     pad = filter_size-1
    pad = int(np.floor(filter_size-1)/2) # ????
    Wp = int(1 + (W + 2*pad - WW)/stride)
    Hp = int(1 + (H + 2*pad - HH)/stride)
    
    HH_m = 2
    WW_m = 2
    stride_m = 2
    Hp_m = int(1 + (Hp - HH_m)/stride_m)
    Wp_m = int(1 + (Wp - WW_m)/stride_m)
    
    self.params['W1'] = weight_scale*np.random.randn(num_filters,C,HH,WW)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale*np.random.randn(num_filters*Wp_m*Hp_m, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    
    if self.use_batchnorm == True:
        self.params['gamma0'] = np.ones(num_filters)
        self.params['beta0'] = np.zeros(num_filters)
        self.params['gamma1'] = np.ones(hidden_dim)
        self.params['beta1'] = np.zeros(hidden_dim)
        self.params['gamma2'] = np.ones(num_classes)
        self.params['beta2'] = np.zeros(num_classes)
        
#         self.params['gamma0'] = np.random.randn(num_filters)
    
        self.bn_param = {'mode': 'test'}
#         self.bn_param[mode] = mode
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax
    
    if self.use_batchnorm == True:
        gamma0 = self.params['gamma0']
        beta0 = self.params['beta0']
        gamma1 = self.params['gamma1']
        beta1 = self.params['beta1']
        gamma2 = self.params['gamma2']
        beta2 = self.params['beta2']
    
    conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
    if self.use_batchnorm == True:
        conv_out, sbatch_cache = spatial_batchnorm_forward(conv_out, gamma0, beta0, self.bn_param)
    rlu1_out, rlu1_cache = relu_forward(conv_out)
    pool_out, pool_cache = max_pool_forward_fast(rlu1_out, pool_param)
    (N,C,H,W) = np.shape(pool_out)
    pool_out_reshape = np.reshape(pool_out,(N,C*H*W))
    aff1_out, aff1_cache = affine_forward(pool_out_reshape, W2, b2)
    if self.use_batchnorm == True:
        aff1_out, aff1_cache = batchnorm_forward(aff1_out, gamma1, beta1, self.bn_param)
    rlu2_out, rlu2_cache = relu_forward(aff1_out)
    scores, aff2_cache = affine_forward(rlu2_out, W3, b3)
    if self.use_batchnorm == True:
        scores, aff2_cache = batchnorm_forward(scores, gamma2, beta2, self.bn_param)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    
    if self.use_batchnorm == True:
        scores,_,_ = batchnorm_backward(scores, cache)
    loss, dx_soft = softmax_loss(scores, y)
    dx_aff2,grads['W3'],grads['b3'] = affine_backward(dx_soft, aff2_cache)
    dx_rlu2 = relu_backward(dx_aff2, rlu2_cache)
    if self.use_batchnorm == True:
        dx_rlu2,_,_ = batchnorm_backward(dx_rlu2, cache)
    dx_aff1,grads['W2'],grads['b2'] = affine_backward(dx_rlu2, aff1_cache)
    dx_aff1_reshape = np.reshape(dx_aff1,(N,C,H,W))
    dx_pool = max_pool_backward_fast(dx_aff1_reshape, pool_cache)
    dx_rlu1 = relu_backward(dx_pool, rlu1_cache)
    if self.use_batchnorm == True:
        dx_rlu1,_,_ = spatial_batchnorm_backward(dx_rlu1, cache)
    _,grads['W1'],grads['b1'] = conv_backward_fast(dx_rlu1, conv_cache)
    
    L2 = lambda x: np.sum(x**2)
    loss += 0.5*self.reg*(L2(W1)+L2(W2)+L2(W3))
    grads['W1'] += self.reg*W1
    grads['W2'] += self.reg*W2
    grads['W3'] += self.reg*W3
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
