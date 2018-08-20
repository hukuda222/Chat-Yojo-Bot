import argparse
import sys

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import train
import pandas as pd
import random
import re
import os


def load_vocab(filename):
    df = pd.read_csv(filename)
    return df['word']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='model data, saved by train_ptb.py')
    parser.add_argument('--primetext', '-p', type=str,
                        default='',
                        help='base text data, used for text generation')
    parser.add_argument('--seed', '-s', type=int, default=50,
                        help='random seeds for text generation')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='number of units')
    parser.add_argument('--sample', type=int, default=10,
                        help='negative value indicates NOT use random choice')
    parser.add_argument('--length', type=int, default=30,
                        help='length of the generated text')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    chainer.config.train = False

    xp = cuda.cupy if args.gpu >= 0 else np

    # load vocabulary
    # vocab = chainer.datasets.get_ptb_words_vocabulary()
    ivocab = load_vocab("words.csv")

    vocab = dict()
    for i, c in enumerate(ivocab):
        vocab[c] = i

    # should be same as n_units , described in train.py
    n_units = args.unit

    lm = train.RNNForLM(len(ivocab), n_units)
    model = L.Classifier(lm)

    serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    model.predictor.reset_state()

    primetext = '<bos>'  # args.primetext
    if isinstance(primetext, six.binary_type):
        primetext = primetext.decode('utf-8')

    if primetext in vocab:
        prev_word = chainer.Variable(xp.array([vocab[primetext]], xp.int32))
    else:
        print('ERROR: Unfortunately ' + primetext + ' is unknown.')
        exit()

    # prob = F.softmax(model.predictor(prev_word))

    sys.stdout.write(primetext)
    prev_word = chainer.Variable(xp.array([vocab[primetext]], xp.int32))

    for j in range(10):
        for i in six.moves.range(args.length):
            prob = F.softmax(model.predictor(prev_word))
            np.random.seed(0)
            if args.sample > 0:
                probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
                probability /= np.sum(probability)
                index = np.random.choice(
                    range(len(probability)), p=probability)
            else:
                index = np.argmax(cuda.to_cpu(prob.data))

            if ivocab[index] == '<eos>':
                sys.stdout.write('<eos>\n\n')
                prev_word = chainer.Variable(
                    xp.array([vocab["<bos>"]], dtype=xp.int32))
                break
            else:
                sys.stdout.write(ivocab[index])

            prev_word = chainer.Variable(xp.array([index], dtype=xp.int32))

        sys.stdout.write('\n')


def get_text(file_name=os.getcwd() + "/../resources/mono_model.npz"):
    xp = np

    ivocab = load_vocab(os.getcwd() + "/words.csv")

    vocab = dict()
    for i, c in enumerate(ivocab):
        vocab[c] = i

    n_units = 650

    lm = train.RNNForLM(len(ivocab), n_units)
    model = L.Classifier(lm)

    serializers.load_npz(file_name, model)

    model.predictor.reset_state()

    return_text = ""
    while return_text != "" or \
            (return_text.find(".co") != -1 or return_text.find("t.") != -1):
        return_text = ""
        prev_word = chainer.Variable(xp.array([vocab["<bos>"]], xp.int32))
        for i in six.moves.range(20):
            prob = F.softmax(model.predictor(prev_word))
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(
                range(len(probability)), p=probability)

            if ivocab[index] == '<eos>':
                break
            else:
                w = ivocab[index]
                j = re.search("[a-z|A-Z|0-9|/|:|_|.]+", w)
                if w.find("/") == -1 and w.find("http") == -1 and \
                   (j is None or j.group(0) != w or len(ivocab[index]) < 6):
                    return_text += ivocab[index]
                else:
                    print("po")

            prev_word = chainer.Variable(xp.array([index], dtype=xp.int32))

    return return_text


if __name__ == '__main__':
    for i in range(100):
        print(get_text())
