import tensorflow as tf 

import numpy as np
import os
#import custom_layers.simple_rnn
import custom_layers.lstm
import tensorflow.keras.layers
import time
from custom_layers.emedding_layer import EmbeddingLayer
from custom_layers.dense_layer import DenseLayer
 
text = open("mt/TaraData/applied.short.train.tgt.txt", "r").read()
subwords = text.split()
sentences= text.split('\n')


vocab = sorted(set(subwords))


sub2idx = {u:i for i, u in enumerate(vocab, start=1)}
sub2idx["<EOS>"] = 0


#array of character representation
idx2sub = np.array(vocab)


#array of the integer representations of the characters
text_as_int = np.array([sub2idx[c] for c in subwords])


list_of_tokens = []
for s in sentences:
  s = s.strip().split()
  list_of_tokens.append([sub2idx[c] for c in s])

sorted_list_of_tokens = list(sorted(list_of_tokens, key = len))

#divide list into lists of size n
def divide_tokens(list, n):
  for i in range(0, len(list), n):  
      yield list[i:i + n] 

#small_list is a test variable that I was able to convert to a tensor becaue the all batches are the same size
small_list = list(divide_tokens(sorted_list_of_tokens[:40], 20))

#all data split into chunks of 20
chunks = list(divide_tokens(sorted_list_of_tokens, 20)) 

#function to pad each batch 
def pad_chunk(list):
  output = []
  for x in list:
    max_length =len(list[-1])
    out = np.zeros(max_length, dtype="int")
    out[:len(x)] = x
    output.append(out)
  return output

padded_chunks = []
for chunk in chunks:
  padded_chunk = pad_chunk(chunk)
  padded_chunks.append(padded_chunk)

seq_length = 20

#how many sequences for corpus
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
sequences = tf.data.Dataset.from_tensor_slices(list(padded_chunks))
#sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


#function to duplicate and shift source/target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

#map the seguences to the function to create input and target
dataset = sequences.map(split_input_target)

#dataset = [split_input_target(x) for x in sequences]


#first example in dataset shifted
for input_example, target_example in dataset.take(1):
  print ('Input data: ', repr(''.join(idx2sub[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2sub[target_example.numpy()])))


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2sub[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2sub[target_idx])))
  
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

#output size
vocab_size = len(vocab)
#dimensions
embedding_dim = 256
#RNN units
rnn_units = 256

emb_layer = EmbeddingLayer(vocab_size, embedding_dim)
dense_layer = DenseLayer(rnn_units, True)
dense_layer_2 = DenseLayer(vocab_size, False)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    EmbeddingLayer(vocab_size, embedding_dim),
    custom_layers.lstm.LSTM(rnn_units),
    DenseLayer(vocab_size, False),
  ])
  return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


for input_example_batch, target_example_batch in dataset.take(1):
    #iterate over custom batches
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    ##already built model, feed it our batches, iterate over batches 


print(model.summary())

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
optimizer = tf.keras.optimizers.Adam()

def train_step(inp, target):
  #looking at gradient (derivative) of loss function and optimize the weights of nn
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


# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))


def generate_text(model, start_string):

  #characters to generate
    num_generate = 1000

  #convert to integer representation
    start_string = start_string.split(' ')
    input_eval = [sub2idx[s] for s in start_string]
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

        text_generated.append(idx2sub[predicted_id])
    
    return (' '.join(text_generated))

print(generate_text(model, start_string=u"h@@ ans@@"))
