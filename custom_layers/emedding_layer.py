from typing import Dict, List, Optional, Any

import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import embedding_ops


class EmbeddingLayer(Layer):
    """
    Embedding layer that maps an integer in [0, input_dim) to an embedding vector of dimensionality output_dim
    Make sure to read the documentation of tensorflow.keras.layers.Layer before reading this code
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 ):
        super(EmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embeddings = None  # type: Optional[tf.Variable]

    def build(self, input_shape: List[Any]):
        # Figure out what the input shape is!
        initializer = initializers.get("uniform")

        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=initializer,
            name='embeddings'
        )
        self.built = True

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

        # this feels like cheating, read docs for explanation
        out = embedding_ops.embedding_lookup(self.embeddings, inputs)
        return out

    def get_config(self) -> Dict[str, Any]:
        config = super(EmbeddingLayer, self).get_config()
        config["input_dim"] = self.input_dim,
        config["output_dim"] = self.output_dim,

        return config


