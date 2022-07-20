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

import sklearn as sk

import os
import csv
import datetime

import IPython
import IPython.display
import seaborn as sns
##

class FilterbankCell1(tf.keras.layers.Layer):
  '''
  Filterbank layer for a single input.  
  '''  
  
  def __init__(self, 
               filters = [1],       
               relax_init = [tf.keras.initializers.constant(0.5)], relax_trainable = True,
               relax_constraint = tf.keras.constraints.MinMaxNorm(min_value=0.01, max_value=0.95),
               sampling_interval = 1, 
               **kwargs):
    super(FilterbankCell1, self).__init__(**kwargs)

    self.num_filterbanks = len(filters)
    self.relax_trainable = relax_trainable
    self.relax_init = relax_init
    self.relax_constraint = relax_constraint
    self.sampling_interval = sampling_interval

    self.state_size = filters
    self.output_size = filters

    self.relax = [[]]*self.num_filterbanks
    for i in range(self.num_filterbanks):
      self.relax[i] = self.add_weight(shape = (1,),
                                      initializer = self.relax_init[i],
                                      trainable = self.relax_trainable,
                                      constraint = self.relax_constraint,
                                      name = 'relax' + str(i+1))  

  def call(self, 
           input_now, output_prev):

    output_now = [[]]*self.num_filterbanks
    for i in range(self.num_filterbanks):
      output_now[i] = generate_dlf_output_now(input_now[:,0:1], output_prev[i], self.relax[i], sampling_interval=self.sampling_interval)

    return output_now, output_now

### Hidden Layer
class HiddenLayer1(tf.keras.layers.Layer):
  
  def __init__(self, hidden_units = 1, hidden_degrees = (1.,),                    
               w_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), w_trainable = True, w_reg = None,
               c_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), c_trainable = True, c_reg = None,
               **kwargs):
      
      self.hidden_units = hidden_units
      
      self.w_init = w_init
      self.w_trainable = w_trainable
      self.w_reg = w_reg
      
      self.hidden_degrees = hidden_degrees
      self.c_init = c_init
      self.c_trainable = c_trainable
      self.c_reg = c_reg
      
      super(HiddenLayer1, self).__init__()

  def build(self, input_shape):
      
    num_inputs = len(input_shape)

    self.w = [[]]*num_inputs
    self.c = [[]]*self.hidden_units

    for i in range(num_inputs):      

      self.w[i] = self.add_weight(shape = (input_shape[i][-1], self.hidden_units), 
                                  initializer = self.w_init, trainable = self.w_trainable,
                                  regularizer = self.w_reg, name = 'w'+str(i+1))
    
    for j in range(self.hidden_units):
      self.c[j] = self.add_weight(shape = (self.hidden_degrees[j], 1),
                                  initializer = self.c_init, trainable = self.c_trainable,
                                  regularizer = self.c_reg, name = 'c_'+str(j+1))
  
  def call(self, input):

      y = input[0] @ self.w[0]
      for i in range(1,len(input)):  
        y += input[i] @ self.w[i]
        
      z = ()    
      for j in range(self.hidden_units):        
        z_j = polynomial(y[:,:,j:j+1], self.c[j])
        z += (z_j,)

      output = tf.concat(z, axis = 2)

      return output

  def get_config(self):
    return {"hidden_units": self.hidden_units,
            "hidden_degrees": self.hidden_degrees,
            "w_init": self.w_init, "w_trainable": self.w_trainable, "w_reg": self.w_reg,
            "c_init": self.w_init, "c_trainable": self.c_trainable, "c_reg": self.c_reg}
###

### Interaction Layer
class InteractionLayer(tf.keras.layers.Layer):
  
  def __init__(self, interaction_units = 1, interaction_degrees = (1.,),                    
               wi_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), wi_trainable = True, wi_reg = None,
               ci_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), ci_trainable = True, ci_reg = None,
               **kwargs):
      
      self.interaction_units = interaction_units
      self.interaction_degrees = interaction_degrees

      self.wi_init = wi_init
      self.wi_trainable = wi_trainable
      self.wi_reg = wi_reg
      
      self.ci_init = ci_init
      self.ci_trainable = ci_trainable
      self.ci_reg = ci_reg
      
      super(InteractionLayer, self).__init__()

  def build(self, input_shape):
      
    num_inputs = len(input_shape)
    
    if num_inputs == 1:

      self.wi_init == tf.initializers.Constant(1.)
      self.wi_trainable = False
      
      self.ci_init == tf.initializers.Constant(1.)
      self.ci_trainable = False
      
      self.interaction_degrees = (1.,)
      
    self.wi = [[]]*num_inputs
    self.ci = [[]]*self.interaction_units

    for i in range(num_inputs):      

      self.wi[i] = self.add_weight(shape = (input_shape[i][-1], self.interaction_units), 
                                  initializer = self.wi_init, trainable = self.wi_trainable,
                                  regularizer = self.wi_reg, name = 'wi'+str(i+1))
    
    for j in range(self.interaction_units):
      self.ci[j] = self.add_weight(shape = (self.interaction_degrees[j], 1),
                                   initializer = self.ci_init, trainable = self.ci_trainable,
                                   regularizer = self.ci_reg, name = 'ci_'+str(j+1))
  
  def call(self, input):

      y = input[0] @ self.wi[0]
      for i in range(1,len(input)):  
        y += input[i] @ self.wi[i]
        
      z = ()    
      for j in range(self.interaction_units):        
        z_j = polynomial(y[:,:,j:j+1], self.ci[j])
        z += (z_j,)

      output = tf.concat(z, axis = 2)

      return output

  def get_config(self):
    return {"interaction_units": self.interaction_units,
            "interaction_degrees": self.interaction_degrees,
            "wi_init": self.wi_init, "wi_trainable": self.wi_trainable, "wi_reg": self.wi_reg,
            "ci_init": self.wi_init, "ci_trainable": self.ci_trainable, "ci_reg": self.ci_reg}
###

### Output Layer
class OutputLayer(tf.keras.layers.Layer):

  def __init__(self, 
               num_outputs = 1, 
               wo_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), wo_trainable = True, wo_reg = None,
               bo_init = tf.keras.initializers.Constant(0.0), bo_trainable = True, bo_reg = None,
               **kwargs):

      self.num_outputs = num_outputs

      if num_outputs == 1:
          wo_init = tf.keras.initializers.Constant(1.)
          wo_trainable = False

      self.wo_init = wo_init
      self.wo_trainable = wo_trainable
      self.wo_reg = wo_reg

      self.bo_init = bo_init
      self.bo_trainable = bo_trainable
      self.bo_reg = bo_reg

      super(OutputLayer, self).__init__(**kwargs)

  def build(self, input_shape):
      
      self.wo = self.add_weight(shape = (input_shape[-1], self.num_outputs),
                                initializer = self.wo_init, 
                                trainable = self.wo_trainable,
                                regularizer = self.wo_reg,
                                name = 'wo')
      
      self.bo = self.add_weight(shape = (self.num_outputs,),
                                initializer = self.bo_init, 
                                trainable = self.bo_trainable,
                                regularizer = self.bo_reg,
                                name = 'bo')

  def call(self, input):

      outputs = input @ self.wo + self.bo

      return outputs

  def get_config(self):
      return {"num_outputs": self.num_outputs,
              "wo_init": self.wo_init, "wo_trainable": self.wo_trainable, "wo_reg": self.wo_reg,
              "bo_init": self.bo_init, "bo_trainable": self.bo_trainable, "bo_reg": self.bo_reg}
###

###
input = tf.keras.Input(shape = (N,1))
 
fb_cell = FilterbankCell(filters = [3, 3],
                         relax_init = [tf.keras.initializers.constant(0.5)]*2)

filterbank_output = tf.keras.layers.RNN(cell = fb_cell,
                                        return_sequences = True, 
                                        return_state = False,
                                        name = 'filterbank_layer')(input)

hidden_layer_output = HiddenLayer1(hidden_units = 2,
                                   hidden_degrees = [2,2],
                                   name = 'hidden_layer')(filterbank_output)

output_layer_output = OutputLayer(num_outputs = 1)(hidden_layer_output)
###

###
class LVN(tf.keras.Model):
  def __init__(self, 
               # inputs, outputs
               num_inputs = 1, num_outputs = 1,
               # filterbank
               filters=[[1]], relax_init=[tf.keras.initializers.constant(0.5)], 
               relax_trainable=[True], relax_constraint=[tf.keras.constraints.MinMaxNorm(min_value=0.01, max_value=0.95)], 
               sampling_interval=1,
               # hidden layer
               hidden_units = [1], hidden_degrees = [[1.0]], 
               w_init=[tf.keras.initializers.random_normal(mean=0.0, stddev=0.01)], w_trainable=[True], w_reg=[None], 
               c_init=[tf.keras.initializers.random_normal(mean=0.0, stddev=0.01)], c_trainable=[True], c_reg=[None],
               # interaction layer
               interaction_units=None, interaction_degrees=None, 
               wi_init=[tf.keras.initializers.random_normal(mean=0.0, stddev=0.01)], wi_trainable=[True], wi_reg=[None], 
               ci_init=tf.keras.initializers.random_normal(mean=0.0, stddev=0.01), ci_trainable=True, ci_reg=None,
               # output layer
               wo_init=tf.keras.initializers.random_normal(mean=0.0, stddev=0.01), wo_trainable=True, wo_reg=None, 
               bo_init=tf.keras.initializers.Constant(0.0), bo_trainable=True, bo_reg=None,
               **kwargs):
    
    super(LVN, self).__init__(**kwargs)

    self.num_inputs = num_inputs
    
    # filterbank and hidden layer
    self.filters = filters
    self.relax_init = relax_init
    self.relax_trainable = relax_trainable
    self.relax_constraint = relax_constraint
    self.sampling_interval = sampling_interval

    self.w_init=w_init
    self.w_trainable=w_trainable
    self.w_reg=w_reg 
    self.c_init=w_init
    self.c_trainable=c_trainable
    self.c_reg=c_reg

    self.filterbanks = [[]]*num_inputs
    for i in range(num_inputs):
       filterbank_cell_i = FilterbankCell1(filters = filters[i],
                                           relax_init = relax_init[i],
                                           relax_trainable = relax_trainable[i], 
                                           relax_constraint = relax_constraint[i], 
                                           sampling_interval = sampling_interval)
       
       self.filterbanks[i] = tf.keras.layers.RNN(cell = filterbank_cell_i,
                                                 return_sequences = True, 
                                                 return_state = False,
                                                 name = 'filterbank_layer' + str(i+1))
       
       self.hidden_layer[i] = HiddenLayer1(hidden_units=self.hidden_units[i], 
                                           hidden_degrees=self.hidden_degrees[i], 
                                           w_init=self.w_init[i], w_trainable=self.w_trainable[i], w_reg=self.w_reg[i], 
                                           c_init=self.c_init[i], c_trainable=self.c_trainable[i], c_reg=self.c_reg[i],
                                           name = 'hidden_layer' + str(i+1))
    
    # interaction layer
    self.interaction_units = interaction_units
    self.interaction_degrees = interaction_degrees
    self.w_init = w_init
    self.w_trainable = w_trainable
    self.w_reg = w_reg
    self.c_init = c_init 
    self.c_trainable = c_trainable
    self.c_reg = c_reg

    self.interaction_layer = None
    if num_inputs > 1:
      self.interaction_layer = InteractionLayer(interaction_units = interaction_units,
                                                interaction_degrees = interaction_degrees,
                                                w_init=w_init, w_trainable=w_trainable, w_reg=w_reg, 
                                                c_init=c_init, c_trainable=c_trainable, c_reg=c_reg,
                                                name = 'interaction_layer')
    
    # output lauyer  
    self.num_outputs = num_outputs
    self.wo_init = wo_init
    self.wo_trainable = wo_trainable
    self.wo_reg = wo_reg
    self.bo_init = bo_init
    self.bo_trainable = bo_trainable
    self.bo_reg = bo_reg
    self.output_layer = OutputLayer(units = num_outputs, 
                                    wo_init = wo_init, 
                                    wo_trainable = wo_trainable, 
                                    wo_reg = wo_reg, 
                                    bo_init = bo_init, 
                                    bo_trainable = bo_trainable, 
                                    bo_reg = bo_reg,
                                    name = 'output_layer')

    def call(self, input):

      filterbank_output = [[]]*self.num_inputs
      for i in range(self.num_inputs):
        filterbank_output[i] = self.filterbank[i](input[:,:,i:i+1])
        hidden_layer_output[i] = self.hidden_layer[i](filterbank_output[i])

      self.interaction_layer_output = None
      if self.num_inputs > 1:
        x = self.interaction_layer(hidden_layer_output)
        self.interaction_layer_output = x
      else:
        x = tf.concat(hidden_layer_output, axis = 2)
      
      output_layer_output = self.output_layer(x)

      self.filterbank_output = filterbank_output
      self.hidden_layer_output = hidden_layer_output
      self.output_layer_output = output_layer_output

      return output_layer_output      
###


# ## import modules
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import sklearn as sk

# import itertools
# import math
# import datetime

# import sys
# sys.modules.keys()

# import sklearn as sk

# import os
# import csv
# import datetime

# import IPython
# import IPython.display
# import seaborn as sns
# ##

# def generate_dlf_output_now(input_now, output_prev, relax, sampling_interval=1):
#   '''
#   Arguments:
#   input_now: scalar
#   output_prev: 1D tensor 
#   relax: scalar
#   sampling_interval: scalar

#   Returns:
#   output_now: tensor
#   '''

#   num_outputs = output_prev.shape[-1]
  
#   sqrt_relax = tf.sqrt(relax)
#   sqrt_1_minus_relax = tf.sqrt(1-relax)

#   # 0th order DLF
#   output_now = sqrt_relax*output_prev[:,0:1] + sampling_interval*sqrt_1_minus_relax*input_now
#   #

#   # ith order DLFs
#   for i in range(1,num_outputs):    
#     output_now_i = sqrt_relax*(output_prev[:,i:i+1] + output_now[:,i-1:i]) - output_prev[:,i-1:i]
#     output_now = tf.concat([output_now, output_now_i], axis=1)
#   #  

#   return output_now

# def polynomial(x, c):
#   '''
#   x = [-1,1]
#   c = [degree,1]
#   '''

#   degree = c.shape[0]
#   pows = tf.range(1,degree+1, dtype='float32')

#   return tf.pow(x, pows) @ c

# class DLF:

#   def __init__(self, units = 1, relax = 0.5, sampling_interval = 1, threshold = 1e-4):
    
#       '''
#       units = number of discrete Laguerre functions.
#       relax = relaxation parameter. Must be between (0,1)
#       sampling_interval = sampling_interval
#       threshold = Minimum absolute value all functions must remain below before stopping
#       '''
      
#       self.units = units
#       self.relax = relax
#       self.sampling_interval = sampling_interval
#       self.threshold = threshold
      
#       sqrt_relax = np.sqrt(relax)
#       sqrt_1_minus_relax = np.sqrt(1-relax)

#       output = generate_dlf_output_now(input_now = 1, output_prev = tf.zeros((1,units)), relax = relax, sampling_interval = sampling_interval)      
#       values = output
      
#       while np.any(tf.abs(output) > threshold):       
#         output = generate_dlf_output_now(input_now = 0, output_prev = output, relax = relax, sampling_interval = sampling_interval)
#         values = tf.concat([values, output], axis = 0)

#       self.values = values
      
#   def conv(self, input):  
#       '''
#       1D convolution of input with each DLF (`self.values`).
#       `input` must be a 1D tensor
#       ''' 
#       values = self.values

#       output = np.zeros((len(input),values.shape[1]))

#       for i,val in enumerate(tf.transpose(values)):
#           output[:,i] = np.convolve(input,val.numpy())[:len(input)]

#       return tf.constant(output)

# ### Leguerre RNN cell
# class FilterbankCell(tf.keras.layers.Layer):

#   def __init__(self, units = (1,),
#                relax_init = (tf.initializers.Constant(0.5),), relax_trainable = True,
#                relax_constraint = tf.keras.constraints.MinMaxNorm(min_value=0.01, max_value=0.95),                  
#                sampling_interval = 1,
#                **kwargs):

#       self.units = units
#       self.output_size = units
#       self.state_size = units
#       self.relax_init = relax_init
#       self.relax_trainable = relax_trainable
#       self.sampling_interval = sampling_interval
#       self.relax_constraint = relax_constraint

#       super(FilterbankCell, self).__init__(**kwargs)
  
#   def build(self, input_shape):
    
#       num_inputs = input_shape[-1]
      
#       self.relax = [[]]*num_inputs
#       for i in range(len(self.units)):
#           self.relax[i] = self.add_weight(shape = (1,),
#                                          initializer = self.relax_init[i],
#                                          trainable = self.relax_trainable,
#                                           constraint = self.relax_constraint,
#                                          name = 'relax' + str(i+1))  
          
#   def call(self, input_now, output_prev):

#     output = [[]]*len(self.units)

#     for i in range(len(self.units)):

#       output[i] = generate_dlf_output_now(input_now[:,i:i+1], output_prev[i], 
#                                           relax = self.relax[i], 
#                                           sampling_interval = self.sampling_interval)
      
#     return output, output

#   def get_config(self):
#       return {"units": self.units,
#               "relax_init": self.relax_init,
#               "relax_trainable": self.relax_trainable,
#               "sampling_interval": self.sampling_interval}  
#   ###

# ### Hidden (polynomial) Layer
# class HiddenLayer(tf.keras.layers.Layer):

#   def __init__(self, units = (1,), hidden_degrees = (1,),                    
#                 w_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), w_trainable = True, w_reg = None,
#                 c_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), c_trainable = True, c_reg = None,
#                 **kwargs):

#       self.units = units

#       self.w_init = w_init
#       self.w_trainable = w_trainable
#       self.w_reg = w_reg
      
#       self.hidden_degrees = hidden_degrees
#       self.c_init = c_init
#       self.c_trainable = c_trainable
#       self.c_reg = c_reg
      
#       super(HiddenLayer, self).__init__()

#   def build(self, input_shape):

#     num_inputs = len(input_shape)
    
#     self.w = [[]]*num_inputs
#     self.c = [[]]*num_inputs

#     for i in range(num_inputs):      

#       self.w[i] = self.add_weight(shape = (input_shape[i][-1], self.units[i]), 
#                                   initializer = self.w_init, trainable = self.w_trainable,
#                                   regularizer = self.w_reg, name = 'w'+str(i+1))
      
#       self.c[i] = [[]]*self.units[i]

#       for j in range(self.units[i]):
#         self.c[i][j] = self.add_weight(shape = ( self.hidden_degrees[i][j], 1),
#                                        initializer = self.c_init, trainable = self.c_trainable,
#                                        regularizer = self.c_reg, name = 'c'+str(i+1)+'_'+str(j+1))

#   def call(self, input):

#       output = [[]]*len(input)

#       for i in range(len(input)): 

#         y_i = input[i] @ self.w[i]

#         z_i = ()
#         for j in range(self.units[i]):
#           z_ij = polynomial(y_i[:,:,j:j+1], self.c[i][j])
#           z_i += (z_ij,)
          
#         output[i] = tf.concat(z_i, axis=2)

#       return output

#   def get_config(self):
#     return {"units": self.units,
#             "hidden_degrees": self.hidden_degrees,
#             "w_init": self.w_init, "w_trainable": self.w_trainable, "w_reg": self.w_reg,
#             "c_init": self.w_init, "c_trainable": self.c_trainable, "c_reg": self.c_reg}
# ###

# ### Interaction Layer
# class InteractionLayer(tf.keras.layers.Layer):
  
#   def __init__(self, units = 1, interaction_degrees = (1.,),                    
#                wi_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), wi_trainable = True, wi_reg = None,
#                ci_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), ci_trainable = True, ci_reg = None,
#                **kwargs):
      
#       self.units = units
      
#       self.wi_init = wi_init
#       self.wi_trainable = wi_trainable
#       self.wi_reg = wi_reg
      
#       self.interaction_degrees = interaction_degrees
#       self.ci_init = ci_init
#       self.ci_trainable = ci_trainable
#       self.ci_reg = ci_reg
      
#       super(InteractionLayer, self).__init__()

#   def build(self, input_shape):
      
#     num_inputs = len(input_shape)
    
#     if num_inputs == 1:

#       self.wi_init == tf.initializers.Constant(1.)
#       self.wi_trainable = False
      
#       self.ci_init == tf.initializers.Constant(1.)
#       self.ci_trainable = False
      
#       self.interaction_degrees = (1.,)
      
#     self.wi = [[]]*num_inputs
#     self.ci = [[]]*self.units

#     for i in range(num_inputs):      

#       self.wi[i] = self.add_weight(shape = (input_shape[i][-1], self.units), 
#                                   initializer = self.wi_init, trainable = self.wi_trainable,
#                                   regularizer = self.wi_reg, name = 'wi'+str(i+1))
    
#     for j in range(self.units):
#       self.ci[j] = self.add_weight(shape = (self.interaction_degrees[j], 1),
#                                    initializer = self.ci_init, trainable = self.ci_trainable,
#                                    regularizer = self.ci_reg, name = 'ci_'+str(j+1))
  
#   def call(self, input):

#       y = input[0] @ self.wi[0]
#       for i in range(1,len(input)):  
#         y += input[i] @ self.wi[i]
        
#       z = ()    
#       for j in range(self.units):        
#         z_j = polynomial(y[:,:,j:j+1], self.ci[j])
#         z += (z_j,)

#       output = tf.concat(z, axis = 2)

#       return output

#   def get_config(self):
#     return {"units": self.units,
#             "interaction_degrees": self.interaction_degrees,
#             "wi_init": self.wi_init, "wi_trainable": self.wi_trainable, "wi_reg": self.wi_reg,
#             "ci_init": self.wi_init, "ci_trainable": self.ci_trainable, "ci_reg": self.ci_reg}
# ###

# ### Output Layer
# class OutputLayer(tf.keras.layers.Layer):

#   def __init__(self, units = 1, 
#               wo_init = tf.keras.initializers.random_normal(mean = 0.0, stddev = 0.01), wo_trainable = True, wo_reg = None,
#               bo_init = tf.keras.initializers.Constant(0.0), bo_trainable = True, bo_reg = None,
#               **kwargs):

#       self.units = units

#       if units == 1:
#           wo_init = tf.keras.initializers.Constant(1.)
#           wo_trainable = False

#       self.wo_init = wo_init
#       self.wo_trainable = wo_trainable
#       self.wo_reg = wo_reg

#       self.bo_init = bo_init
#       self.bo_trainable = bo_trainable
#       self.bo_reg = bo_reg

#       super(OutputLayer, self).__init__(**kwargs)

#   def build(self, input_shape):
      
#       self.wo = self.add_weight(shape = (input_shape[-1], self.units),
#                                 initializer = self.wo_init, 
#                                 trainable = self.wo_trainable,
#                                 regularizer = self.wo_reg,
#                                 name = 'wo')
      
#       self.bo = self.add_weight(shape = (self.units,),
#                                 initializer = self.bo_init, 
#                                 trainable = self.bo_trainable,
#                                 regularizer = self.bo_reg,
#                                 name = 'bo')

#   def call(self, input):

#       outputs = input @ self.wo + self.bo

#       return outputs

#   def get_config(self):
#       return {"units": self.units,
#               "wo_init": self.wo_init, "wo_trainable": self.wo_trainable, "wo_reg": self.wo_reg,
#               "bo_init": self.bo_init, "bo_trainable": self.bo_trainable, "bo_reg": self.bo_reg}
#   ### 

# class LVN(tf.keras.layers.Layer):
  
#   def __init__(self,
#                # filterbank parameters
#                filters=(1,), relax_init = (tf.initializers.Constant(0.5),), relax_trainable = True, 
#                relax_constraint = tf.keras.constraints.MinMaxNorm(min_value=0.05, max_value=0.95), 
#                sampling_interval = 1,
#                # hidden layer parameters
#                hidden_units=(1,), hidden_degrees=(1,), 
#                w_init=tf.keras.initializers.random_normal(mean=0.0, stddev=0.01), 
#                w_trainable=True, w_reg=None, 
#                c_init=tf.keras.initializers.random_normal(mean=0.0, stddev=0.01), 
#                c_trainable=True, c_reg=None,
#                # interaction layer parameters
#                interaction_units=1, interaction_degrees=(1.,), 
#                wi_init=tf.keras.initializers.random_normal(mean=0.0, stddev=0.01), 
#                wi_trainable=True, wi_reg=None, 
#                ci_init=tf.keras.initializers.random_normal(mean=0.0, stddev=0.01), 
#                ci_trainable=True, ci_reg=None,
#                # output layer parameters
#                num_outputs = 1, 
#                wo_init = tf.keras.initializers.random_normal(mean=0.0, stddev=0.01), 
#                wo_trainable = True, wo_reg = None,
#                bo_init = tf.keras.initializers.Constant(0.0), 
#                bo_trainable = True, bo_reg = None,
#                **kwargs):
    
#       super(LVN, self).__init__(**kwargs)

#       self.num_inputs = len(filters)

#       ## filterbank layer
#       self.filters = filters
#       self.relax_init = relax_init
#       self.relax_trainable = relax_trainable
#       self.relax_constraint = relax_constraint
#       self.sampling_interval = sampling_interval

#       self.filterbank_cell = FilterbankCell(units = filters, 
#                                       relax_init = relax_init, 
#                                       relax_trainable = True, 
#                                       relax_constraint = relax_constraint,
#                                       sampling_interval = 1)
      
#       self.filterbank = tf.keras.layers.RNN(cell = self.filterbank_cell, 
#                                       return_sequences = True,
#                                       return_state = False,
#                                       unroll = False,
#                                       name = 'filterbank')
#       ##

#       ## hidden layer
#       self.hidden_units = hidden_units
#       self.hidden_degrees = hidden_degrees
#       self.w_init = w_init
#       self.w_trainable = w_trainable
#       self.w_reg = w_reg
#       self.c_init = c_init
#       self.c_trainable = c_trainable
#       self.c_reg = c_reg

#       self.hidden_layer = HiddenLayer(units = hidden_units,
#                                               hidden_degrees = hidden_degrees,
#                                               name = 'hidden_layer')
#       ##

#       ## interaction layer (ignore if only 1 stimulus)
#       if self.num_inputs == 1:
#         self.interaction_units = None
#         self.interaction_degrees = None
#         self.wi_init = None
#         self.wi_trainable = None
#         self.wi_reg = None
#         self.ci_init = None
#         self.ci_trainable = None
#         self.ci_reg = None
#         self.interaction_layer = None
#       else:
#         self.interaction_units = interaction_units
#         self.interaction_degrees = interaction_degrees
#         self.wi_init = wi_init
#         self.wi_trainable = wi_trainable
#         self.wi_reg = wi_reg
#         self.ci_init = ci_init
#         self.ci_trainable = ci_trainable
#         self.ci_reg = ci_reg
#         self.interaction_layer = InteractionLayer(units = interaction_units,
#                                                   interaction_degrees = interaction_degrees,
#                                                   name = 'interaction_layer')
#       ##

#       ## output layer
#       self.num_outputs = num_outputs
#       self.wo_init = wo_init
#       self.wo_trainable = wo_trainable
#       self.wo_reg = wo_reg
#       self.bo_init = bo_init
#       self.bo_trainable = bo_trainable
#       self.bo_reg = bo_reg
      
#       self.output_layer = OutputLayer(units=num_outputs, 
#                                       wo_init=wo_init, 
#                                       wo_trainable=wo_trainable, 
#                                       wo_reg=wo_reg, 
#                                       bo_init=bo_init, 
#                                       bo_trainable=bo_trainable, 
#                                       bo_reg=bo_reg,
#                                       name = 'output_layer')
#       ##

#   def call(self, input):
    
#     filterbank_output = self.filterbank(input)

#     hidden_layer_output = self.hidden_layer(filterbank_output)

#     if self.num_inputs == 1:
#       x = hidden_layer_output
#     else:
#       x = self.interaction_layer(hidden_layer_output)

#     output1 = self.output_layer(x)

#     return output1

#   def get_config(self):
#     return {'filters': self.filters,
#       'relax_init': self.relax_init,
#       'relax_trainable': self.relax_trainable,
#       'relax_constraint': self.relax_constraint,
#       'sampling_interval': self.sampling_interval,
#       'hidden_units': self.hidden_units,
#       'hidden_degrees': self.hidden_degrees,
#       'w_init': self.w_init,
#       'w_trainable': self.w_trainable,
#       'w_reg': self.w_reg,
#       'c_init': self.c_init,
#       'c_trainable': self.c_trainable,
#       'c_reg': self.c_reg,
#       'interaction_units': self.interaction_units,
#       'interaction_degrees': self.interaction_degrees,
#       'wi_init': self.wi_init,
#       'wi_trainable': self.wi_trainable,
#       'wi_reg': self.wi_reg,
#       'ci_init': self.ci_init,
#       'ci_trainable': self.ci_trainable,
#       'ci_reg': self.ci_reg,
#       'num_outputs': self.num_outputs,
#       'wo_init': self.wo_init,
#       'wo_trainable': self.wo_trainable,
#       'wo_reg': self.wo_reg,
#       'bo_init': self.bo_init,
#       'bo_trainable': self.bo_trainable,
#       'bo_reg': self.bo_reg}



# ### NMSE  
# def nmse(y_true,y_pred):
#   return tf.reduce_mean(tf.pow(y_true-y_pred,2)) / tf.reduce_mean(y_true**2)  
# ###
