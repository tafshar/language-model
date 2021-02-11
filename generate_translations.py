import tensorflow as tf
import json

import sys
import numpy as np
from custom_layers.mt_model import MtModel
from custom_layers.attention import Attention
import constants as constants
from vocabulary import Vocabulary


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

# output size
vocab_size = len(sub2idx)
# dimensions
embedding_dim = 256
# RNN units
rnn_units = 256

model = MtModel(vocab_size, embedding_dim, rnn_units)
checkpoint_path = "training_checkpoints/ckpt_9"
model.load_weights(checkpoint_path)

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
    #decoder_input = tf.squeeze(tf.expand_dims([constants.SOS_ID], 1), -1)
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

        
#print(generate_text(model, source_sentence=u"the cl@@ er@@ k of the house"))


print("\nEnter source sentence: ")
x = input()
print(generate_text(model, source_sentence=x))
