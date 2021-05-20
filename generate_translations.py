import tensorflow as tf
import json

import sys
import numpy as np
from typing import List, Optional, Any, Generator, Dict, Tuple, TextIO
from custom_layers.mt_model import MtModel
from custom_layers.attention import Attention
import constants as constants
from vocabulary import Vocabulary


src = open("mt/TaraData/applied.medium.train.src.txt", "r").read()
trg = open("mt/TaraData/applied.medium.train.tgt.txt", "r").read()
sub2idx_json = open("enfr_vocab_med_txt", "r").read()

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
embedding_dim = 64
# RNN units
rnn_units = 64

model = MtModel(vocab_size, embedding_dim, rnn_units)
checkpoint_path = "training_checkpoints/ckpt_49"
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

    encoder_out, hidden_state, carry_state = model.call_encoder(input_eval)
    dec_hidden = model.init_hidden(1)
    dec_carry = model.init_hidden(1)

    temperature = 1.0

    decoder_input = [constants.SOS_ID]
    translation_id = []
    attention = []

    for i in range(num_generate):
        decoder_input_tf = tf.constant(decoder_input, dtype=tf.int32)
        dec_out, dec_hidden, dec_carry, attn_weights = model.call_decoder(decoder_input_tf, encoder_out, dec_hidden, dec_carry)
        attention.append(attn_weights)
        
        #dec_out = tf.squeeze(dec_out, 0)

        dec_out = dec_out / temperature
        predicted_id = tf.math.argmax(dec_out, axis=-1)[0].numpy()
        decoder_input = predicted_id
        predicted_id = tf.squeeze(predicted_id, 0)

        current_subword = idx2sub[int(predicted_id)]
        translation_id.append(current_subword)
        
        if predicted_id == 0:
          tf.stack(attention)
          result = ' '.join(translation_id)
          return result, attention

    tf.stack(attention)
    result = ' '.join(translation_id)
    return result, attention


def attention_matrix_to_nematus(source_units: List[str],
                                target_units: List[str],
                                attention_matrix: np.array,
                                line_count: int) -> str:
    # create nematus alignment string as described in and used for https://github.com/M4t1ss/SoftAlignments
    lines = []

    source_len = len(source_units)
    # last target index is the <EOS> token in the attention_matrix
    target_max_len = attention_matrix.shape[0] - 1
    target_len = min(len(target_units), target_max_len)
    target_string, sentence_string = " ".join(target_units), " ".join(source_units)

    # we do not have a score, so set it to 0.0
    score = 0.0
    header = "{} ||| {} ||| {} ||| {} ||| {} {}".format(line_count, target_string, score,
                                                        sentence_string, source_len, target_len)
    lines.append(header)

    # also look at the attention for EOS
    for tgt_id in range(target_len + 1):
        attention_matrix_test = attention_matrix.squeeze()
        attention_string = (" ".join([str(np_float) for np_float in attention_matrix_test[tgt_id]]))
        lines.append(attention_string)
    # One empty line at the end of each matrix according to the nematus specification
    lines.append("")

    nematus_string = "\n".join(lines)
    nematus_string += "\n"

    return nematus_string

        

#print(generate_text(model, source_sentence=u"le dis@@ cours d@@ u tr@@ ô@@ ne"))

src ="le président de él@@ ection"
targ, att = generate_text(model, source_sentence=src)
source_units = src.split(" ")
target_units = targ.split(" ")
nematus_string = attention_matrix_to_nematus(source_units, target_units, np.array(att), 0)

with open("valid.nematus", 'w') as f:
  f.write(nematus_string)



#print("\nEnter source sentence: ")
#x = input()
#print(generate_text(model, source_sentence=x))
