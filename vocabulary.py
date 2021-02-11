import tensorflow as tf
import numpy as np
from typing import Iterable, Dict, Set, List, Optional
import json
import constants as constants


class Vocabulary:

    def __init__(self, sub2idx: Dict[str, int]):
        self.sub2idx = sub2idx
        self.idx2sub = {v: k for (k, v) in self.sub2idx.items()}

    def to_disk(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.sub2idx, f)

    def create_vocab(source, target):

        src = open(source, "r").read()
        trg = open(target, "r").read()

        ##################q
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
        sub2idx[constants.SOS] = constants.SOS_ID
        sub2idx[constants.EOS] = constants.EOS_ID


        return Vocabulary(sub2idx)
