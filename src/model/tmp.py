import argparse
import collections
import pickle
import glob

import numpy as np
import pandas as pd

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, serializers
from chainer import reporter
from chainer import training
from chainer.training import extensions

def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--serializeInterval', '-s', type=int, default=500,
                        help='Interval for serialize model.')
    parser.add_argument('--batchsize', '-b', type=int, default=30,
                        help='Number of images in each mini-batch.')
    parser.add_argument('--epoch', '-e', type=int, default=2,
                        help='Number of sweeps over the dataset to train.')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU).')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result.')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot.')
    parser.add_argument('--unit', '-u', type=int, default=400,
                        help='Number of units.')
    parser.add_argument('--modelPath', '-m', default='../../../Data/Chat-Yojo-Bot/Model/',
                        help='Model dir to store serialized model.')
    parser.add_argument('--conversationFname', '-c', default='../../../Data/Chat-Yojo-Bot/Corpus/tokenizedConversationAskFM.json',
                        help='Conversation file name.')
    parser.add_argument('--baseDataDir', '-d', default='../../../Data/',
                        help='Base directory for learning.')
    args = parser.parse_args()

    conversationDf = pd.read_json("../../../../Data/Chat-Yojo-Bot/Corpus/ConversationAskFM.json", encoding = "utf-8") # Make DataFrame from json.

    print(conversationDf[10:12])

if __name__ == '__main__':
    main()
