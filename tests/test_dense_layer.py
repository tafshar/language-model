from custom_layers.dense_layer import DenseLayer
import numpy as np
import tensorflow as tf


#def test_dense():
#    units = 64
#    dense_layer = DenseLayer(units,False)
#    #taking input (2, 5)
#    #(batch_size, input_dim)

    #output is getting (batch size, units(output))
#    assert dense_layer(np.zeros((2,5))).shape == (2, units)
    #weight is taking (batch size, input dim)
#    assert dense_layer.w.shape == (5, units)
#    assert dense_layer.b.shape == (units,)

def test_learning_with_single_input():
    units = 3

    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    dl = DenseLayer(units, True)
    dl.build([(2,3)])
    breakpoint()


    for _ in range(5):
        with tf.GradientTape() as tape:
            # shape output vector will be (output_dim,)
            input_data = np.random.rand(3,2)
            output = dl(input_data)
            #breakpoint()
            output_mean = tf.reduce_mean(output)
            #define error/loss
            output_mean = tf.reduce_mean(output)
            loss = tf.math.abs(output_mean - 27.0)
            grads = tape.gradient(loss, dl.trainable_variables)
            optimizer.apply_gradients(zip(grads, dl.trainable_variables))
        new_output = dl(input_data)
        