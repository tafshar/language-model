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
from custom_layers.attention import Attention
import constants as constants
from vocabulary import Vocabulary
import json



########
# Process training data and load vocab
# ######### 

src = open("mt/TaraData/applied.med.train.tgt.txt", "r").read()
trg = open("mt/TaraData/applied.med.train.tgt.txt", "r").read()
sub2idx_json = open("vocab_med_trg.txt", "r").read()


sub2idx = json.loads(sub2idx_json)

# reverse mapping
idx2sub = {v: k for k, v in sub2idx.items()}

sentences_src = src.split('\n')
sentences_trg = trg.split('\n')

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
    batch = np.zeros((batch_size, max_len + 2), dtype=np.int32)
    for i in range(batch_size):
      example = [constants.SOS_ID] + sorted_list_of_examples[start_idx+i]
      batch[i, :len(example)] = example
    list_of_batches.append(batch)
  
  return list_of_batches

#all data split into chunks of 20
src_data = [src_tuple[0] for src_tuple in sorted_list_of_tokens]
trg_data = [trg_tuple[1] for trg_tuple in sorted_list_of_tokens]

src_batches = create_batches(src_data, 20)
trg_batches = create_batches(trg_data, 20)

seq_length = 20


#function to duplicate and shift source/target
def split_input_target(source, target):
    input_text = source[:, :-1]
    target_text = target[:, 1:]
    return input_text, target_text


##################
#### Create source and target input datasets
##################


merged_dataset = list(zip(src_batches, trg_batches))


input_example_batch, target_example_batch = merged_dataset[-1]
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

src_input_example_batch, src_target_example_batch = merged_dataset[-1]

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

optimizer = tf.keras.optimizers.Adam()

def train_step(inp, target):
  loss = 0.0
  with tf.GradientTape() as tape:
    enc_output, enc_hidden, enc_carry = model.call_encoder(inp)
    dec_hidden = enc_hidden
    dec_carry = enc_carry
    for targidx in range(target.shape[1]-1):
      dec_out, dec_hidden, dec_carry, _ = model.call_decoder(target[:, targidx], enc_output, dec_hidden, dec_carry)
      loss += tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
          target[:, targidx+1], dec_out, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  #update parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

EPOCHS = 10

print('Number of batches per epoch: {}'.format(len(merged_dataset)))
for epoch in range(EPOCHS):
  start = time.time()
  sum_loss = 0

  random.shuffle(merged_dataset) 
  for (batch_n, (inp, target)) in enumerate(merged_dataset):
    loss = train_step(inp, target)
    sum_loss += loss

    if batch_n +1 == len(merged_dataset):
      template = 'Epoch {} Loss {}'
      avg_loss = sum_loss / batch_n
      print(template.format(epoch+1, avg_loss))
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
    input_eval = [constants.SOS_ID] + [sub2idx[s] for s in source_sentence] + [constants.EOS_ID]
    input_eval = tf.expand_dims(input_eval, 0)

    final_encoder_out, final_hidden_state, final_carry_state = model.call_encoder(input_eval)

    temperature = 1.0

    decoder_input = [constants.SOS_ID]
    translation_id = []

    for i in range(num_generate):
        decoder_input_tf = tf.constant(decoder_input, dtype=tf.int32)
        dec_out, dec_hidden, dec_carry, attn_weights = model.call_decoder(decoder_input_tf, final_encoder_out, final_hidden_state, final_carry_state)
        
        #dec_out = tf.squeeze(dec_out, 0)

        dec_out = dec_out / temperature
        predicted_id = tf.math.argmax(dec_out, axis=-1)[0].numpy()
        decoder_input = predicted_id
        predicted_id = tf.squeeze(predicted_id, 0)

        current_subword = idx2sub[int(predicted_id)]
        translation_id.append(current_subword)
        
        if predicted_id == 0:
          return (' '.join(translation_id))

    return(' '.join(translation_id))
        

print(generate_text(model, source_sentence=u"le dis@@ cours d@@ u tr@@ ô@@ ne"))
