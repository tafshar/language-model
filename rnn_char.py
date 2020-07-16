import tensorflow as tf 

import numpy as np
import os
import time
from custom_layers.emedding_layer import EmbeddingLayer
from custom_layers.dense_layer import DenseLayer
 
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')


vocab = sorted(set(text))



#mapping from string to numerical representation (indices)
#this is the dictionary mapping every unique characted in the text to an index
char2idx = {u:i for i, u in enumerate(vocab)}

#array of character representation
idx2char = np.array(vocab)


#array of the integer representations of the characters
text_as_int = np.array([char2idx[c] for c in text])



# The maximum length sentence we want for a single input in characters
seq_length = 100

#how many sequences for corpus
examples_per_epoch = len(text)//(seq_length+1)


# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

####item is of length 101, take 5 items
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))
    print(len(item))

#function to duplicate and shift source/target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

#map the seguences (our text batched into desired sequence length) to the function to create input and target
dataset = sequences.map(split_input_target)

#first example in dataset shifted
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
  
BATCH_SIZE = 64
BUFFER_SIZE = 10000
#Question: shuffle within buffer since the tf.data is meant to work with infinite sequences
#Question: how is training batch size determined, what is ideal?

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

#output size
vocab_size = len(vocab)
#dimensions
embedding_dim = 256
#RNN units
rnn_units = 1204


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    EmbeddingLayer(vocab_size, embedding_dim),
    DenseLayer(rnn_units, 1),
    DenseLayer(vocab_size, 0)
  ])
  return model

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

print(model.summary())

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

optimizer = tf.keras.optimizers.Adam()

def train_step(inp, target):
  #looking at gradient (derivative) of loss fucntion and optimize the weights of nn
  #calculating the loss within training loop so we can see how each prediciton and loss changes per epoch/batch
  with tf.GradientTape() as tape:
    predictions = model(inp)
    #forward calc
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  #what calculate gradients with respect to the model
  grads = tape.gradient(loss, model.trainable_variables)
  #update parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

# Training step
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  # initializing the hidden state at the start of every epoch
  # initally hidden is None
  hidden = model.reset_states()

  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  # saving (checkpoint) the model every 5 epochs
  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))

def generate_text(model, start_string):

  #characters to generate
    num_generate = 1000

  #convert to integer representation
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # Question: remove the batch dimension?
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

 
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])
    
    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))
