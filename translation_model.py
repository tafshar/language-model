import tensorflow as tf 
import tensorflow.keras.layers

from typing import List
import random
import numpy as np
import os
import custom_layers.lstm
import time
from custom_layers.emedding_layer import EmbeddingLayer
from custom_layers.dense_layer import DenseLayer
from custom_layers.mt_model import MtModel
 
src = open("mt/TaraData/applied.short.train.src.txt", "r").read()
trg = open("mt/TaraData/applied.short.train.src.txt", "r").read()

##################
#### source and target processing
##################

combined = src + trg
combined_subwords = combined.split()

subwords_src = src.split()
sentences_src= src.split('\n')

subwords_trg = trg.split()
sentences_trg= trg.split('\n')


vocab = sorted(set(combined_subwords))


sub2idx = {u:i for i, u in enumerate(vocab, start=2)}
# End of sentence
EOS_ID = 0
sub2idx["<EOS>"] = EOS_ID

# Start of sentence
SOS_ID = 1
sub2idx["<SOS>"] = SOS_ID

# reverse mapping
idx2sub = {v: k for k, v in sub2idx.items()}


#array of the integer representations of the characters
text_as_int = np.array([sub2idx[c] for c in combined_subwords])


src_tokens = []
for s in sentences_src:
  s = s.strip().split()
  src_tokens.append([sub2idx[c] for c in s])

trg_tokens = []
for s in sentences_trg:
  s = s.strip().split()
  trg_tokens.append([sub2idx[c] for c in s])

################# 
##### Create batches
#################

def sort_lists_by_length(src_data: List[List[int]], tgt_data: List[List[int]]):
    data_tuples = [(a, b) for a, b in zip(src_data, tgt_data)]
    return sorted(data_tuples, key=lambda x: len(x[1]))

sorted_list_of_tokens = sort_lists_by_length(src_tokens, trg_tokens)

#divide list into lists of size n
def create_batches(sorted_list_of_examples: List[List[int]], batch_size: int):
  list_of_batches: List[np.array] = []
 
  for start_idx in range(0, len(sorted_list_of_examples) - batch_size, batch_size):
    max_len = max(len(x) for x in sorted_list_of_examples[start_idx:start_idx+batch_size])
    # TODO: we may get an incomplete batch at the end
    # The first token should always be SOS_ID and the last token should always be the EOS_ID/0, that's why we use max_len+2
    batch = np.zeros((batch_size, max_len + 2), dtype=np.int32)
    for i in range(batch_size):
      example = [SOS_ID] + sorted_list_of_examples[start_idx+i]
      batch[i, :len(example)] = example
    list_of_batches.append(batch)
  
  return list_of_batches

#all data split into chunks of 20
src_data = [src_tuple[0] for src_tuple in sorted_list_of_tokens]
trg_data = [trg_tuple[1] for trg_tuple in sorted_list_of_tokens]

src_batches = create_batches(src_data, 20)
trg_batches = create_batches(trg_data, 20)

seq_length = 20

examples_per_epoch = len(combined)//(seq_length+1)


#function to duplicate and shift source/target
def split_input_target(chunk):
    input_text = chunk[:, :-1]
    target_text = chunk[:, 1:]
    return input_text, target_text


##################
#### Create source and target input datasets
##################

src_dataset = [split_input_target(x) for x in src_batches]
trg_dataset = [split_input_target(x) for x in trg_batches]


input_example_batch, target_example_batch = src_dataset[-1]
input_example = input_example_batch[0]
target_example = target_example_batch[0]

  
BATCH_SIZE = 64
BUFFER_SIZE = 10000

#output size
vocab_size = len(sub2idx)
#dimensions
embedding_dim = 256
#RNN units
rnn_units = 256

##################
#### Train step
################## 

model = MtModel(vocab_size, embedding_dim, rnn_units)

src_input_example_batch, src_target_example_batch = src_dataset[-1]
example_batch_predictions = model(input_example_batch, target_example_batch)


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

optimizer = tf.keras.optimizers.Adam()

def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp, target)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  #update parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  random.shuffle(src_dataset) 
  for (batch_n, (inp, target)) in enumerate(src_dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))


print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))

##################
#### Generate translation
##################

def generate_text(model, source_sentence):

  #characters to generate
    num_generate = 1000

    source_sentence = source_sentence.split(' ')
    input_eval = [SOS_ID] + [sub2idx[s] for s in source_sentence]
    input_eval = tf.expand_dims(input_eval, 0)

    final_memory_state, final_carry_state = model.get_final_encoder_states(input_eval)

    temperature = 1.0

    decoder_input = [[SOS_ID]]
    translation_id = []

    for i in range(num_generate):
        decoder_input_tf = tf.constant(decoder_input, dtype=tf.int32)
        predictions, final_memory_state, final_carry_state = model.next_step_decoder(decoder_input_tf, final_memory_state, final_carry_state)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        decoder_input = [[predicted_id]]
       
        translation_id.append(idx2sub[predicted_id])
        
        if predicted_id == 0:
          return (' '.join(translation_id))

        

print(generate_text(model, source_sentence=u"sit@@ ting re@@ su@@ m@@ ed"))
