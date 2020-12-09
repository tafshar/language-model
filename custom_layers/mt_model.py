import tensorflow as tf
import numpy as np
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer


class MtModel(tf.keras.Model):
    
    def __init__(self, vocab_size: int, emb_dim: int, rnn_units: int):
        super().__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, emb_dim)
        
        self.encoder_lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=False, return_state=True)
        self.decoder_lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, stateful = False)
        self.decoder_output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, src_batch, tgt_batch):
        """ tgt_batch: [batch_size, seq_len] """
        
        src_embedding = self.embedding_layer(src_batch)
        tgt_embedding = self.embedding_layer(tgt_batch)

        src_encoder, final_memory_state, final_carry_state = self.encoder_lstm(src_embedding)
        tgt_decoder, tgt_final_memory_state, tgt_final_carry_state = self.decoder_lstm(tgt_embedding,
                                        initial_state=(final_memory_state, final_carry_state))

        decoder_output = self.decoder_output_layer(tgt_decoder)
        return decoder_output

    def get_final_encoder_states(self, src_batch: tf.Tensor) -> [tf.Tensor, tf.Tensor]: 
        final_src_emb = self.embedding_layer(src_batch)
        final_encoder, final_memory_state, final_carry_state = self.encoder_lstm(final_src_emb)
        return final_memory_state, final_carry_state


    def next_step_decoder(self, tgt_example, final_memory_state, final_carry_state) -> tf.Tensor:
        tgt_emb = self.embedding_layer(tgt_example)
        final_tgt_decoder, final_tgt_memory_state, final_tgt_carry_state = self.decoder_lstm(tgt_emb, initial_state=(final_memory_state, final_carry_state))
        final_decoder_output = self.decoder_output_layer(final_tgt_decoder)
        return final_decoder_output, final_tgt_memory_state, final_tgt_carry_state


