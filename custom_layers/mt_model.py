import tensorflow as tf
import numpy as np
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer


class MtModel(Layer):
    
    def __init__(self, vocab_size: int, emb_dim: int, rnn_units: int):
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, emb_dim)
        
        self.encoder_lstm = tf.keras.layers.LSTM(rnn_units, return_sequences = False, return_state=True)
        self.decoder_lstm = tf.keras.layers.LSTM(rnn_units, return_state=True)
        self.decoder_output_layer = tf.keras.layers.Dense(vocab_size)

    def forward(self, src_batch, tgt_batch):
        src_embedding = self.embedding_layer(src_batch)
        tgt_embedding = self.embedding_layer(tgt_batch)

        src_encoder, final_memory_state, final_carry_state = self.encoder_lstm(src_embedding)
        tgt_decoder = self.decoder_lstm(tgt_embedding, initial_state = final_memory_state)

        decoder_output = self.decoder_output_layer(tgt_decoder)
        return decoder_output
        



        











#def build_encoder(vocab_size, embedding_dim, rnn_units, batch_size):
#  model = tf.keras.Sequential([
#    EmbeddingLayer(vocab_size, embedding_dim),
#    tf.keras.layers.LSTM(rnn_units, return_state=True),
#  ])
#  return model
#######LSTM EX
##lstm = tf.keras.layers.LSTM(256, return_state=True)
##output, final_memory_state = lstm(src_dataset[0])
##print(final_memory_state.shape)
#def build_decoder(vocab_size, embedding_dim, rnn_units, batch_size):
#  model = tf.keras.Sequential([
#    EmbeddingLayer(vocab_size, embedding_dim),
#    custom_layers.lstm.LSTM(rnn_units),
#    #tf.keras.layers.LSTM(rnn_units), 
#    # WITH CALL ARGUMENT initial state = final_memory_state
#  ])
#  return model

