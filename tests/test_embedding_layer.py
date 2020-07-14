from custom_layers.emedding_layer import EmbeddingLayer
import tensorflow as tf


def test_embedding_with_list_of_inputs():
    input_dim = 40  # size of vocab
    output_dim = 64  # size of resulting embedding vector

    embedding_layer = EmbeddingLayer(input_dim, output_dim)
    output_vector = embedding_layer([1, 5, 7])
    assert output_vector.shape == (3, output_dim)


def test_embedding_with_single_input():
    input_dim = 40  # size of vocab
    output_dim = 64  # size of resulting embedding vector

    embedding_layer = EmbeddingLayer(input_dim, output_dim)
    output_vector = embedding_layer(12)
    assert output_vector.shape == (output_dim, )


def test_learning_with_single_input():
    input_dim, output_dim = 1, 2

    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    embedding_layer = EmbeddingLayer(input_dim, output_dim)

    with tf.GradientTape() as tape:
        # shape output vector will be (output_dim,)
        output_vector = embedding_layer(0)
        # bring the output closer to 42
        errors = tf.abs(output_vector - 42.0)
        summed_errors = tf.math.reduce_sum(errors)
        variables_to_train = embedding_layer.trainable_variables
        grads = tape.gradient(summed_errors, variables_to_train)
        optimizer.apply_gradients(zip(grads, variables_to_train))

    new_output = embedding_layer(0)
    new_summed_errors = tf.math.reduce_sum(tf.abs(new_output - 42.0))

    assert new_summed_errors < summed_errors

