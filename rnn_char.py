import tensorflow as tf 

import numpy as np
import os
import time
 
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

print ('Length of text: {} characters'.format(len(text)))
#print(text[:1000])

vocab = sorted(set(text))
print('{} unique charcters'.format(len(vocab)))


#mapping from string to numerical representation (indices)
#this is the dictionary mapping every unique characted in the text to an index
char2idx = {u:i for i, u in enumerate(vocab)}

#array of character representation
idx2char = np.array(vocab)


#array of the integer representations of the characters
text_as_int = np.array([char2idx[c] for c in text])
print(len(text_as_int))


print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


# The maximum length sentence we want for a single input in characters
seq_length = 100

#how many sequences for corpus
examples_per_epoch = len(text)//(seq_length+1)


# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
print(type(char_dataset))


for i in char_dataset.take(5):
    print(text_as_int[i.numpy()])

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
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
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

#questions about this
print(model.summary())

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#Question: what dimension size/shape does axis -1 return
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

#print(idx2char[sampled_indices])

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


#QUESTION: losses
example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10
#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):

  #characters to generate
    num_generate = 1000

  #convert to integer representation
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 2.0

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
