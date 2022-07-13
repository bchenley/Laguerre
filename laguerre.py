## import modules
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk

import itertools
import math
import datetime

import sys
sys.modules.keys()
##

class DLF:

  def __init__(self, units = 1, relax = 0.5, samp_interval = 1, thresh = 1e-4):
    
      '''
      units = number of discrete Laguerre functions.
      relax = relaxation parameter. Must be between (0,1)
      samp_interval = sampling_interval
      thresh = Minimum absolute value all functions must remain below before stopping
      '''
      
      self.units = units
      self.relax = relax
      self.samp_interval = samp_interval
      self.thresh = thresh
      
      sqrt_relax = np.sqrt(relax)
      sqrt_1_minus_relax = np.sqrt(1-relax)

      values = np.empty((1,units))

      step = 0; loop = True
      while loop:      
            for i in range(units):
                if i ==0:
                    if step == 0:
                        values[step,i] = samp_interval*sqrt_1_minus_relax*1
                    else:
                        values[step,i] = sqrt_relax*values[step-1,i] # + samp_interval*sqrt_1_minus_relax*1
                else:
                    if step == 0:
                        values[step,i] = sqrt_relax*values[step,i-1]
                    else:
                        values[step,i] = sqrt_relax*values[step-1,i] + sqrt_relax*values[step,i-1] - values[step-1,i-1]
            if np.all(np.abs(values[-1,:]) < thresh):
                loop = False
            else:
                step +=1
                values = np.append(values,np.zeros((1,units)),axis = 0)

      self.values = values
      self.steps = tf.linspace(0,step,step+1)

  def conv(self, input):  
      '''
      1D convolution of input with each DLF (`self.values`).
      `input` must be a 1D tensor
      ''' 
      values = self.values

      output = np.zeros((len(input),values.shape[1]))

      for i,val in enumerate(tf.transpose(values)):
          output[:,i] = np.convolve(input,val.numpy())[:len(input)]

      return tf.constant(output)

### Leguerre Rnn cell
class FilterbankCell(tf.keras.layers.Layer):

  def __init__(self, units = (1,),
              relax_init = (0.5,), relax_trainable = True,                    
              samp_interval = 1, 
              relax_name = ('relax1',),
              **kwargs):

      self.units = units
      self.output_size = units
      self.state_size = units
      self.relax_init = relax_init
      self.relax_trainable = relax_trainable
      self.samp_interval = samp_interval
      self.relax_name = relax_name
     
      self.relax = ()
      for i in range(len(self.units)):
          self.relax += (tf.Variable(initial_value = relax_init[i],
                                    trainable = relax_trainable,
                                    name = relax_name[i]),)      

      super(FilterbankCell, self).__init__(**kwargs)

  def call(self, input, state):

    prev_output = state

    T = self.samp_interval

    output = [[]]*len(self.units)

    for i in range(len(self.units)):

      input_i = input[:,i:i+1]  
      prev_output_i = prev_output[i]
      
      relax_i = self.relax[i]
      sqrt_relax_i = tf.math.sqrt(relax_i)
      sqrt_1_minus_relax_i = tf.math.sqrt(1 - relax_i)

      output_i = sqrt_relax_i*prev_output_i[:,0:1] + T*sqrt_1_minus_relax_i*input_i

      for j in range(1,self.units[i]):        

        output_ij = sqrt_relax_i*prev_output_i[:,j:j+1] + sqrt_relax_i*output_i[:,j-1:j] - prev_output_i[:,j-1:j]

        output_i = tf.concat((output_i,output_ij),axis = 1)

      output[i] = output_i

    return output, output

  def get_config(self):
      return {"units": self.units,
              "relax_init": self.relax_init,
              "relax_trainable": self.relax_trainable,
              "samp_interval": self.samp_interval,
              "relax_name": self.relax_name}  
  ###

### Hidden (polynomial) Layer
class HiddenLayer(tf.keras.layers.Layer):

  def __init__(self, units = (1,), degree = (1,),                    
                w_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.05), w_trainable = True, w_reg = None, w_name = ('w1',),
                c_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.05), c_trainable = True, c_reg = None, c_name = ('c1',),
                **kwargs):

      self.units = units

      self.w_init = w_init
      self.w_trainable = w_trainable
      self.w_reg = w_reg
      self.w_name = w_name

      self.degree = degree
      self.c_init = c_init
      self.c_trainable = c_trainable
      self.c_reg = c_reg
      self.c_name = c_name
      
      super(HiddenLayer, self).__init__()

  def build(self, input_shape):

    self.w = ()
    self.c = ()

    for i in range(len(input_shape)):          
      self.w += (self.add_weight(shape = (input_shape[i][-1], self.units[i]), 
                                initializer = self.w_init, trainable = self.w_trainable,
                                regularizer = self.w_reg, name = self.w_name[i]),)

      self.c += (self.add_weight(shape = (self.degree[i], self.units[i]), 
                              initializer = self.c_init, trainable = self.c_trainable,
                              regularizer = self.c_reg, name = self.c_name[i]),)

  def call(self, input):

      degree = self.degree
      w = self.w
      c = self.c

      output = ()

      for i in range(len(input)):      

        y_i = input[i] @ w[i]

        z_i = 0.
        for q in range(degree[i]):
          z_i += tf.math.multiply(tf.pow(y_i,q+1) , c[i][q:q+1,:])

        output += (z_i,)

      return output

  def get_config(self):
    return {"units": self.units,
            "degree": self.degree,
            "w_init": self.w_init, "w_trainable": self.w_trainable, "w_reg": self.w_reg, "w_name": self.w_name,
            "c_init": self.w_init, "c_trainable": self.c_trainable, "c_reg": self.c_reg, "c_name": self.c_name}
###

### Interaction Layer
class InteractionLayer(tf.keras.layers.Layer):

  def __init__(self, units = 1, degree = 1,                    
                wi_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.05), wi_trainable = True, wi_reg = None, wi_name = 'wi1',
                ci_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.05), ci_trainable = True, ci_reg = None, ci_name = 'ci',
                **kwargs):

      self.units = units

      self.wi_init = wi_init
      self.wi_trainable = wi_trainable
      self.wi_reg = wi_reg
      self.wi_name = wi_name

      self.degree = degree
      self.ci_init = ci_init
      self.ci_trainable = ci_trainable
      self.ci_reg = ci_reg
      self.ci_name = ci_name

      super(InteractionLayer, self).__init__()

  def build(self, input_shape):

      self.wi = ()

      for i in range(len(input_shape)):

        self.wi += (self.add_weight(shape = (input_shape[i][-1], self.units), 
                                    initializer = self.wi_init, trainable = self.wi_trainable,
                                    regularizer = self.wi_reg, name = self.wi_name[i]),)

      self.ci = self.add_weight(shape = (self.degree, self.units), 
                                initializer = self.ci_init, trainable = self.ci_trainable,
                                regularizer = self.ci_reg, name = self.ci_name)

  def call(self, input):

      degree = self.degree
      w = self.wi
      c = self.ci

      y = 0.
      for i in range(len(input)):

        y_i = input[i] @ w[i]
        y += y_i

      output = 0.
      for q in range(degree):
        output += tf.math.multiply(tf.pow(y,q+1), c[q:q+1,:])

      return output

  def get_config(self):
    return {"units": self.units,
            "degree": self.degree,
            "wi_init": self.wi_init, "wi_trainable": self.wi_trainable, "wi_reg": self.wi_reg, "wi_name": self.wi_name,
            "ci_init": self.wi_init, "ci_trainable": self.ci_trainable, "ci_reg": self.ci_reg, "ci_name": self.ci_name}
  ###

### Output Layer
class OutputLayer(tf.keras.layers.Layer):

  def __init__(self, units = 1, 
              wo_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.05), wo_trainable = True, wo_reg = None, wo_name = 'wo',
              bo_init = tf.keras.initializers.Constant(0.0), bo_trainable = True, bo_reg = None, bo_name = 'bo',
              **kwargs):

      self.units = units

      if units == 1:
          wo_init = tf.keras.initializers.Constant(1.)
          wo_trainable = False

      self.wo_init = wo_init
      self.wo_trainable = wo_trainable
      self.wo_reg = wo_reg
      self.wo_name = wo_name

      self.bo_init = bo_init
      self.bo_trainable = bo_trainable
      self.bo_reg = bo_reg
      self.bo_name = bo_name

      self.bo = tf.Variable(initial_value = 0.0,
                            trainable = bo_trainable,
                            name = 'bo')

      super(OutputLayer, self).__init__(**kwargs)

  def build(self, input_shape):

      self.wo = self.add_weight(shape = (input_shape[-1], self.units),
                                initializer = self.wo_init, 
                                trainable = self.wo_trainable,
                                regularizer = self.wo_reg,
                                name = self.wo_name)

  def call(self, input):
      
      wo = self.wo
      bo = self.bo
      
      outputs = input @ wo + bo

      return outputs

  def get_config(self):
      return {"units": self.units,
              "wo_init": self.wo_init, "wo_trainable": self.wo_trainable, "wo_reg": self.wo_reg, "wo_name": self.wo_name,
              "bo_init": self.bo_init, "bo_trainable": self.bo_trainable, "bo_reg": self.bo_reg, "bo_name": self.bo_name}
  ### 

### NMSE  
def nmse(y_true,y_pred):
  return tf.reduce_mean(tf.pow(y_true-y_pred,2)) / tf.reduce_mean(y_true**2)  
###
