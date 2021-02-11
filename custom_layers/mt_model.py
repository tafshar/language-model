import tensorflow as tf
import numpy as np
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer
from custom_layers.attention import Attention


class MtModel(tf.keras.Model):
    
    def __init__(self, vocab_size: int, emb_dim: int, rnn_units: int):
        super().__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, emb_dim)
        
        self.encoder_lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
        self.decoder_lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, stateful = False)
        self.decoder_output_layer = tf.keras.layers.Dense(vocab_size)
        self.attention = Attention(rnn_units)
        ####Question: verify attention dims (output dimension?)

    def call_encoder(self, src_batch):
        src_embedding = self.embedding_layer(src_batch)
        src_encoder, final_memory_state, final_carry_state = self.encoder_lstm(src_embedding)
        return src_encoder, final_memory_state, final_carry_state


    def call_decoder(self, tgt_batch, encoder_out, hidden_state, carry_state):
        tgt_embedding = self.embedding_layer(tgt_batch)
        context, attn_weights = self.attention(encoder_out, hidden_state)
        #context_concat = tf.concat((tf.expand_dims(context, len(tgt_batch[1])), tgt_embedding), -1)
        context_concat = tf.concat((context, tgt_embedding), -1)
        context_concat = tf.expand_dims(context_concat, 1)
        tgt_decoder, tgt_final_memory_state, tgt_final_carry_state = self.decoder_lstm(context_concat,
                                        initial_state=(hidden_state, carry_state))
        decoder_output = self.decoder_output_layer(tgt_decoder)
        return decoder_output, tgt_final_memory_state, tgt_final_carry_state, attn_weights

    def init_hidden(self):
        return tf.zeros(shape=(self.src_batch, self.rnn_units))


    def get_final_encoder_states(self, src_batch: tf.Tensor) -> [tf.Tensor, tf.Tensor]: 
        final_src_emb = self.embedding_layer(src_batch)
        final_encoder, final_memory_state, final_carry_state = self.encoder_lstm(final_src_emb)
        return final_memory_state, final_carry_state


    def next_step_decoder(self, tgt_example, final_memory_state, final_carry_state) -> tf.Tensor:
        tgt_emb = self.embedding_layer(tgt_example)
        final_tgt_decoder, final_tgt_memory_state, final_tgt_carry_state = self.decoder_lstm(tgt_emb, initial_state=(final_memory_state, final_carry_state))
        final_decoder_output = self.decoder_output_layer(final_tgt_decoder)
        return final_decoder_output, final_tgt_memory_state, final_tgt_carry_state


