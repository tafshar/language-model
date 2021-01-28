import os
import argparse
from vocabulary import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser("Create vocabulary from files")
    parser.add_argument("--source", required=True, type=str, help="path to src file")
    parser.add_argument("--target", required=True, type=str, help="path to trg file")
    parser.add_argument("--vocab_path", required=True, type=str, help="path of vocabulary")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src = args.source
    trg = args.target

    vocab = Vocabulary.create_vocab(src, trg)
    #breakpoint()
    vocab.to_disk(args.vocab_path)
    print(f"Wrote vocabulary to {args.vocab_path}.")
