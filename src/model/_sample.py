import sys
import time
import argparse

import numpy as np

import chainer
from chainer import cuda
from chainer import optimizers

from _DataProcessing import *
from _Model import *


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##########################################################
    parser.add_argument(
        "--gpu",
        dest="gpu",
        default=-1,
        type=int,
        help=("GPU ID [int] (negative value indicates CPU)"))
    parser.add_argument(
        "-B",
        "--batch-size",
        dest="batch_size",
        default=16,
        type=int,
        help="mini batch size [int] default=256")
    parser.add_argument(
        "-SE",
        "--save-epoch",
        dest="save_epoch",
        default=5,
        type=int,
        help="epoch size [int] for saving model default=15")
    ##########################################################

    ##########################################################
    parser.add_argument(
        "--enc-vocab-file",
        dest="encVocabFile",
        default="../../../../Data/Chat-Yojo-Bot/Vocab/enc_vocab_Twitter.pickle",
        help="filename of encoder (input)-side vocabulary")
    parser.add_argument(
        "--dec-vocab-file",
        dest="decVocabFile",
        default="../../../../Data/Chat-Yojo-Bot/Vocab/dec_vocab_Twitter.pickle",
        help="filename of decoder (output)-side vocabulary")
    parser.add_argument(
        "--enc-data-file",
        dest="encDataFile",
        default="../../../../Data/Chat-Yojo-Bot/Corpus/enc_text_Twitter.pickle",
        help="filename of encoder (input)-side data for training")
    parser.add_argument(
        "--dec-data-file",
        dest="decDataFile",
        default="../../../../Data/Chat-Yojo-Bot/Corpus/dec_text_Twitter.pickle",
        help="filename of decoder (output)-side data for trainig")
    parser.add_argument(
        "--saved-model-dir",
        dest="savedModelDir",
        default="../../../../Data/Chat-Yojo-Bot/Model/",
        help="dirname of directory for saving model")
    ##########################################################

    ##########################################################
    parser.add_argument(
        "--lrate",
        dest="lrate",
        default=0.1,
        type=float,
        help="learning rate [float] default=1.0")
    parser.add_argument(
        "--dropout-rate",
        dest="dropout_rate",
        default=0.05,
        type=float,
        help="dropout rate [float] default=0.3")
    parser.add_argument(
        "--gradient-clipping",
        dest="grad_clip",
        default=0.5,
        type=float,
        help="gradient clipping threshold [float] default=3.0")
    ##########################################################

    ##########################################################
    parser.add_argument(
        "-DE",
        "--embed-dim_enc",
        dest="eDim_enc",
        default=256,
        type=int,
        help=("dimensions of embedding layers in both encoder [int] default=512"))
    parser.add_argument(
        "-DD",
        "--embed-dim_dec",
        dest="eDim_dec",
        default=256,
        type=int,
        help=("dimensions of embedding layers in both decoder [int] default=256"))
    parser.add_argument(
        "-H",
        "--hidden-dim",
        dest="hDim",
        default=256,
        type=int,
        help="dimensions of all hidden layers [int] default=512")
    parser.add_argument(
        "-N",
        "--num-rnn-layers",
        dest="n_layers",
        default=1,
        type=int,
        help=("number of RNN (LSTM) layers in both encoder/decoder [int] default=2"))
    ##########################################################

    ##########################################################
    parser.add_argument(
        "--random-seed",
        dest="seed",
        default=2723,
        type=int,
        help="random seed [int] default=2723")
    ##########################################################

    args = parser.parse_args()

    assert (args.save_epoch >= 1), "Models should be saved per epoch which over 1!"

    if args.gpu >= 0: cuda.get_device(args.gpu).use()

    prepD = PrepareData()
    encoderVocab = prepD.readVocab(args.encVocabFile, input_side=True)
    decoderVocab = prepD.readVocab(args.decVocabFile, input_side=False)
    encSentLenDict = prepD.makeSentenceLenDict(args.encDataFile, encoderVocab, input_side=True)
    decSentLenDict = prepD.makeSentenceLenDict(args.decDataFile, decoderVocab, input_side=False)

    trainData = prepD.makeBatch4Train(encSentLenDict, decSentLenDict, batch_size=args.batch_size)
    # [([arr(el1), ...], [arr(dl1), ...]),
    #  ([...], [...]),
    #  ...
    # ]

    ED = EncoderDecoder(encoderVocab, decoderVocab, trainData, args)
    ED.initModel()
    ED.loadModel(args.savedModelDir)
    ED.setToGPUs()

    optimizer = optimizers.Adam(args.lrate)
    optimizer.setup(ED.model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.00000001))

    for epoch in range(1, 101):
        begin = time.time()

        # random.shuffle(trainData)
        ED.train(optimizer, trainData, epoch)

        end = time.time()

        if epoch % args.save_epoch == 0:
            ED.saveModel(args.savedModelDir, args.save_epoch)

        sys.stdout.write("# Total Time for epoch: {}\n".format(end - begin))
