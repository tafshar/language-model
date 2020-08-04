from typing import Dict, List, Any

import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer


class LSTM(Layer):
    """
    Equation 54 and 55 in [mt and seq2seq lecture chapter 6]
    (http://phontron.com/class/mtandseq2seq2018/assets/slides/mt-fall2018.chapter6.pdf)
    """
    def __init__(self,
                 units: int,
                 ):
        #
        super().__init__(dynamic=True)
        self.units = units

    def build(self, input_shape: List[Any]):
        """ Input shape should be (batch_dim, sequence_length, input_dimension) """
        assert len(input_shape) == 3
        self.input_dim = input_shape[-1]

        # Figure out what the input shape is!
        glorot_init = initializers.get("glorot_uniform")
        zero_init = initializers.get("zeros")

        self.w_xu = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=glorot_init,
            name='wxh'
        )
        self.w_hu = self.add_weight(
            shape=(self.units, self.units),
            initializer=glorot_init,
            name='whh'
        )
        self.b_u = self.add_weight(
            shape=(self.units),
            initializer=zero_init,
            name='bias'
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        batch_dim = input_shape[0]
        sequence_length = input_shape[1]

        return (batch_dim, sequence_length, self.units)

    def call(self,
             inputs: tf.Tensor,
             **kwargs):
        """
        Converts a multiple of ints to multiple embedding vectors
        :param inputs: Tensor of ints containing the ids to lookup in the embedding matrix
        :param kwargs: Any additional keyword arguments to match parent function
        :return: Embeddings of dimensionality self.output_dim
        """
        # one dim case
        # out = self.embeddings[inputs[0]]
        # out = self.embeddings[inputs[0]]
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]

        outputs = []  # to keep track of output values

        h = tf.zeros(shape=(batch_size, self.units))  # initial cell state
        c = tf.zeros(shape=(batch_size, self.units))  # initial hidden state
        step = 0
        while step < sequence_length:
            x = inputs[:, step, :]
            u = tf.tanh(tf.matmul(x, self.w_xu) + tf.matmul(h, self.w_hu) + self.b_u)
            i = 0.0  # TODO
            f = 1.0  # TODO
            o = 1.0  # TODO
            c = i * u + f * c
            h = o * tf.tanh(c)

            outputs.append(h)
            step += 1

        outputs_stacked = tf.stack(outputs, axis=1)
        return outputs_stacked

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["units"] = self.units
        return config
