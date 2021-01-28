import tensorflow as tf
import numpy as np
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
    
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, encoder_out, hidden):
        #shape of encoder_out : batch_size, seq_length, hidden_dim (16, 10, 1024)
        #shape of encoder_hidden : batch_size, hidden_dim (16, 1024)
        
        hidden = tf.expand_dims(hidden, axis=1) #out: (16, 1, 1024)
        
        score = self.V(tf.nn.tanh(self.W1(encoder_out) + \
                                  self.W2(hidden))) #out: (16, 10, 1)
        
        attn_weights = tf.nn.softmax(score, axis=1)
        
        context =  attn_weights * encoder_out #out: ((16,10,1) * (16,10,1024))=16, 10, 1024
        context = tf.reduce_sum(context, axis=1) #out: 16, 1024
        return context, attn_weights