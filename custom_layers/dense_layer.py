from typing import Dict, List, Optional, Any

import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer


class DenseLayer(Layer):

    def __init__(self,
                 input_dim: int,
                 units: int,
                 #activation_function: Optional[str],
                 ):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.units = units

    def build(self, input_shape: List[Any]):
    #state of weight and bias
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units)),
                            
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,)),
            trainable=True)

    def call(self, inputs):  
        #Given an input x, output y= func(W*x+b)
        return tf.matmul(inputs, self.w) + self.b


