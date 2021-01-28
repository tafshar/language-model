import tensorflow as tf
import numpy as np

# Initialize and define Variables somehow
one_init = tf.constant_initializer(value=1.0)
zero_init = tf.constant_initializer(value=0.0)
W = tf.Variable(initial_value=one_init(shape=(2, 3)))
b = tf.Variable(zero_init(shape=(2,1)))
# Input
x = np.array([[1.0], [2.0], [3.0]], dtype='float32')

# Calculate gradients with tensorflow
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
with tf.GradientTape() as tape:
    #Tanh(W*x + 3 * b)
    y = W @ x + 3 * b
    z = tf.math.tanh(y)
    tf_grads = tape.gradient(z, [W, b, y])
tf_grad_W, tf_grad_b, tf_grad_y = tf_grads

# Calculate gradients by hand
# gradients dz/dx = dz/dy * dy/dx
# derivative of tanh(x) = 1/(cosh(x))^2 

# Gradient with respect to W: dz/dW = dz/dy * dy/dW = 1/(cosh(x))^2 * W
grad_y = 1.0 / tf.math.square(tf.math.cosh(y))
grad_W = grad_y @ tf.transpose(x)
# Gradient wrt. b: dz/db = dz/dy * dy/db = 1/(cosh(x))^2 * 3.0
grad_b = grad_y * 3.0

# Update by hand
W_2 = W - grad_W
b_2 = b - grad_b
# Update by tensorflow
optimizer.apply_gradients(zip([tf_grad_W, tf_grad_b], [W, b]))

