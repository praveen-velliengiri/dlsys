"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    #number of outputs in the gradient matches the
    #number of inputs to the operation
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)

class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
      return array_api.power(a, b)
    
    '''
    f = a^b
    f'a = b a^b-1
    f'b = a^b log (a)
    '''
    def gradient(self, out_grad, node):
      base, powarg = node.inputs #(a, b)
      fg = powarg * power(base, powarg-1)
      sg = power(base, powarg) * log(base)
      return (fg * out_grad, sg * out_grad)

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
      return array_api.power(a, self.scalar)

    # f = a^b
    # f'a = b * a^b-1
    def gradient(self, out_grad, node):
      basearg,  = node.inputs
      fg = mul_scalar(power_scalar(basearg, self.scalar-1), self.scalar)
      return (fg * out_grad, )


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
      return array_api.divide(a, b)

    '''
      f = (a/b)
      da = 1/b
      db = - a/b^2
    '''
    def gradient(self, out_grad, node):
      numtensor, dentensor = node.inputs
      np_ones_tensor = array_api.ones(dentensor.shape)
      #we don't want gradient of this ones tensor.
      ones_tensor = Tensor.make_const(np_ones_tensor, requires_grad = False)
      fg = divide(ones_tensor, dentensor)
      sg = negate(divide(numtensor, power_scalar(dentensor, 2)))
      return fg * out_grad, sg * out_grad

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
      return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
      return divide_scalar(out_grad, self.scalar)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
      ndim = array_api.ndim(a)
      if self.axes is None:
        self.axes = [ndim-1, ndim-2]
      
      full_axes = array_api.arange(ndim)
      full_axes[self.axes[0]] = self.axes[1]
      full_axes[self.axes[1]] = self.axes[0]
      return array_api.transpose(a, axes=full_axes)
      
    #tricky part: incoming gradient is of the transposed tensor
    #so we need to make flowing gradient to be consistent with
    #the input shape.
    def gradient(self, out_grad, node):
      return transpose(out_grad, self.axes)

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
      return array_api.reshape(a, self.shape)

    #similar to transpose
    #incoming gradient is for a reshape tensor.
    #we need to undo the reshape.
    def gradient(self, out_grad, node):
      input_shape = node.inputs[0].shape
      return reshape(out_grad, input_shape)

def reshape(a, shape):
    return Reshape(shape)(a)


def identify_broadcasted_axes(broadcast_shape, original_shape):
  orig = (1,) * (len(broadcast_shape) - len(original_shape)) + original_shape
  axes = []
  for i, (o, b) in enumerate(zip(orig, broadcast_shape)):
    if o == 1 and b > 1:
      axes.append(i)
  return axes

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
      return array_api.broadcast_to(a, self.shape)

    '''
      so the incoming gradient is of the broadcasted shape,
      we need to find the broadcasted axes, and sum the incoming
      gradient along those axes.
      after that we need to reshape that is to remove the unit dimensions.
    '''
    def gradient(self, out_grad, node):
      axes_to_sum = identify_broadcasted_axes(out_grad.shape, node.inputs[0].shape)
      return reshape(summation(out_grad, tuple(axes_to_sum)), node.inputs[0].shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

def gradient(self, out_grad, node):
    shape_to_reshape = list(node.inputs[0].shape)  # copy input shape
    


    for i in unpack_axes:
        shape_to_reshape[i] = 1

    # reshape so broadcast is possible
    broadcasted_grad = reshape(out_grad, tuple(shape_to_reshape))
    return broadcast_to(broadcasted_grad, node.inputs[0].shape)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
      return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
      shape_to_reshape = list(node.inputs[0].shape) #explicit copy with list.
      #unpack_axes = range(len(shape_to_reshape)) if self.axes is None else self.axes
      # normalize axes into iterable
      
      if self.axes is None:
        unpack_axes = range(len(shape_to_reshape))
      elif isinstance(self.axes, int):
        unpack_axes = [self.axes]
      else:
        unpack_axes = self.axes
      
      for i in unpack_axes:
        shape_to_reshape[i] = 1
      
      #reshape so that broadcast is possible.
      broadcasted_grad = reshape(out_grad, shape_to_reshape)
      return broadcast_to(broadcasted_grad, node.inputs[0].shape)
      

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
      return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
      a,b = node.inputs

      agrad = matmul(out_grad, transpose(b))
      bgrad = matmul(transpose(a), out_grad)

      if len(a.shape) < len(agrad.shape):
        #tuple is necessary it seems.
        sum_axes = tuple(array_api.arange(len(agrad.shape) - len(a.shape)))
        agrad = summation(agrad, sum_axes)
      
      if len(b.shape) < len(bgrad.shape):
        sum_axes = tuple(array_api.arange(len(bgrad.shape) - len(b.shape)))
        bgrad = summation(bgrad, sum_axes)
      return agrad, bgrad

def matmul(a, b):
    return MatMul()(a, b)

class Negate(TensorOp):
    def compute(self, a):
      return array_api.multiply(a, -1)

    def gradient(self, out_grad, node):
      np_ones_tensor = array_api.ones(out_grad.shape)
      #we don't want gradient of this ones tensor.
      ones_tensor = Tensor.make_const(np_ones_tensor, requires_grad = False)
      return out_grad * negate(ones_tensor)

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
      return array_api.log(a)

    #log(a) = 1/a
    def gradient(self, out_grad, node):
      a = node.inputs[0]
      return divide(out_grad, a)

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
      return array_api.exp(a)

    #differentiate(e^a) = e^a
    def gradient(self, out_grad, node):
        return multiply(out_grad, exp(node.inputs[0]))
        raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
      return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
      x = node.inputs[0]
      return out_grad * (x.realize_cached_data() > 0)




def relu(a):
    return ReLU()(a)

