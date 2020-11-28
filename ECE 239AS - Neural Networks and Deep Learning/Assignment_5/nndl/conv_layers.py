import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  (N,C,H,W) = x.shape
  (F,_,HH,WW) = w.shape

  Wp = int(1 + (W + 2*pad - WW)/stride)
  Hp = int(1 + (H + 2*pad - HH)/stride)
  out = np.zeros((N,F,Hp,Wp))
  
  for n in np.arange(N):
      xpad = np.pad(x[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
      for f in np.arange(F):
          for j in np.arange(Hp):
              hi = j*stride
              hf = hi + HH
              for i in np.arange(Wp):
                  wi = i*stride
                  wf = wi + WW
                  out[n,f,j,i] = np.sum(np.multiply(xpad[:,hi:hf,wi:wf],w[f,:,:,:])) + b[f]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #

  (_,_,H,W) = x.shape
  (_,_,HH,WW) = w.shape
  (_,_,Hout,Wout) = dout.shape
  
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  
  for n in np.arange(N):
      xpad = np.pad(x[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
      dxpad = np.zeros_like(xpad)
      for f in np.arange(F):
          for j in np.arange(Hout):
              hi = j*stride
              hf = hi + HH
              for i in np.arange(Wout):
                  wi = i*stride
                  wf = wi + WW
                  dxpad[:,hi:hf,wi:wf] += w[f,:,:,:]*dout[n,f,j,i]
                  dw[f,:,:,:] += xpad[:,hi:hf,wi:wf]*dout[n,f,j,i]
                  db[f] += dout[n,f,j,i]
                
      dx[n,:,:,:] = dxpad[:,pad:H+pad,pad:W+pad]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  (N,C,H,W) = x.shape
  PH = pool_param['pool_height']
  PW = pool_param['pool_width']
  stride = pool_param['stride']

  Wp = int(1 + (W - PW)/stride)
  Hp = int(1 + (H - PH)/stride)
  out = np.zeros((N,C,Hp,Wp))
  
  for n in np.arange(N):
      for j in np.arange(Hp):
          hi = j*stride
          hf = hi + PH
          for i in np.arange(Wp):
              wi = i*stride
              wf = wi + PW
              out[n,:,j,i] = np.max(np.reshape(x[n,:,hi:hf,wi:wf],(C,PW*PH)),axis=1)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  
  (N,C,H,W) = x.shape
  PW = pool_width
  PH = pool_height
    
  Wp = int(1 + (W - PW)/stride)
  Hp = int(1 + (H - PH)/stride)
  dx = np.zeros_like(x)

  for n in np.arange(N):
      for j in np.arange(Hp):
          hi = j*stride
          hf = hi + PH
          for i in np.arange(Wp):
              wi = i*stride
              wf = wi + PW
              for c in np.arange(C):
                  max = np.max(np.reshape(x[n,c,hi:hf,wi:wf],(1,PW*PH))) 
                  xa = x[n,c,hi:hf,wi:wf]
                  mask = xa == max*np.ones_like(xa)
                  dx[n,c,hi:hf,wi:wf] += mask*dout[n,c,j,i] 
 
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #

  (N, C, H, W) = np.shape(x)
  x_re = np.transpose(x,(0,2,3,1))
  xr = np.reshape(x_re, (N*H*W, C))
  outr, cache = batchnorm_forward(xr, gamma, beta, bn_param)
  out_re = np.reshape(outr, (N, H, W, C))
  out = np.transpose(out_re,(0,3,1,2))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
    
  (N, C, H, W) = np.shape(dout)
  dout_re = np.transpose(dout,(0,2,3,1))
  doutr = np.reshape(dout_re, (N*H*W, C))
  dxr, dgamma, dbeta = batchnorm_backward(doutr, cache)
  dx_re = np.reshape(dxr, (N, H, W, C))
  dx = np.transpose(dx_re,(0,3,1,2))

  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta