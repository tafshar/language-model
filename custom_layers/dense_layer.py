from typing import Dict, List, Optional, Any

import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer


class DenseLayer(Layer):

    def __init__(self,
                 units: int,
                 _use_activation_function: bool,
                 ):
        super(DenseLayer, self).__init__()
        self.units = units
        self._use_activation_function = _use_activation_function
    

    def build(self, input_shape: List[Any]):
    #state of weight and bias
        #w_init = tf.random_normal_initializer()
        w_init = tf.zeros_initializer()
        #w_init = [[1.0, 1.0, 1.0,],[2.0,2.0,2.0]]
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units)),                           
            trainable=True, name="weight")
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,)),
            trainable=True, name="bias")
        self.build = True

    def call(self, inputs):  
        #breakpoint()
        #Given an input x, output y= func(W*x+b)
        output = tf.matmul(inputs, self.w) + self.b
        if self._use_activation_function:
           return tf.keras.activations.sigmoid(output)
        else:
           return output


