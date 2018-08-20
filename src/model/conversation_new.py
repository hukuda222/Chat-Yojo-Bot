import argparse
import collections
import pickle
import MeCab
import sys
import os

import numpy as np
import zenhan as zh

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, serializers
from chainer import reporter
from chainer import training

# from chainer.training import extensions

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs

def norm_embed(bn, xs):
    x_len = [len(x.data) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = bn(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs

class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_vocab, n_units):
        super(Seq2seq, self).__init__(
            embed_x=L.EmbedID(n_vocab, n_units),
            embed_y=L.EmbedID(n_vocab, n_units),
            encoder=L.NStepLSTM(n_layers, n_units, n_units, 0.3),
            decoder=L.NStepLSTM(n_layers, n_units, n_units, 0.3),
            W=L.Linear(n_units, n_vocab),
            bnormDec=L.BatchNormalization(n_units),
        )
        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, *inputs):
        xs = inputs[:len(inputs) // 2]
        ys = inputs[len(inputs) // 2:]

        eos = self.xp.array([0], 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # Initial hidden variable and cell variable.
        zero = self.xp.zeros((self.n_layers, batch, self.n_units), 'f')
        hx, cx, _ = self.encoder(zero, zero, exs)

        # sliHx = F.get_item(hx, (slice(self.n_layers - 2 + 1, self.n_layers - 1 + 1)))
        # sliCx = F.get_item(cx, (slice(self.n_layers - 2 + 1, self.n_layers - 1 + 1)))

        _, _, os = self.decoder(hx, cx, eys)
        # os = norm_embed(self.bnormDec, os)
        loss = F.softmax_cross_entropy(
            self.W(F.concat(os, axis=0)), F.concat(ys_out, axis=0))

        reporter.report({'loss': loss.data}, self)
        return loss

    def translate(self, xs, max_length=10):
        batch = len(xs)
        with chainer.no_backprop_mode():
            exs = sequence_embed(self.embed_x, xs)
            # Initial hidden variable and cell variable
            zero = self.xp.zeros((self.n_layers, batch, self.n_units), 'f')
            h, c, _ = self.encoder(zero, zero, exs, train=False)

            # sliH = F.get_item(h, (slice(self.n_layers - 2 + 1, self.n_layers - 1 + 1)))
            # sliC = F.get_item(c, (slice(self.n_layers - 2 + 1, self.n_layers - 1 + 1)))

            ys = self.xp.zeros(batch, 'i')
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = chainer.functions.split_axis(
                    eys, batch, 0, force_tuple=True)
                h, c, ys = self.decoder(h, c, eys, train=False)
                # ys = norm_embed(self.bnormDec, ys)
                cys = chainer.functions.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = np.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

def convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [to_device(x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = to_device(concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return tuple(
        to_device_batch([x for x, _ in batch]) +
        to_device_batch([y for _, y in batch]))

def conversation(words, model, dictionary, id2wd):
    # words = ['意外と', 'この' , '問題', '見掛け倒し', 'だっ', 'た', 'ゾ']
    x = model.xp.array([dictionary[w] for w in words], 'i')
    ys = model.translate([x])[0]
    words = [id2wd[y] for y in ys]
    rep = ''.join(words)
    return rep



parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
parser.add_argument('--modelPath', '-m', default=os.getcwd()+'/../resources/seq_2_seq-68.model',
                    help='Path to the model')
parser.add_argument('--unit', '-u', type=int, default=400,
                    help='Number of units')
parser.add_argument('--baseDataDir', '-d', default='../resources',
                    help='Base directory for learning')
args = parser.parse_args()

dicFname = os.getcwd()+"/../resources/partial_vocab_dict.pickle"
with open(dicFname, "rb") as f:
    if sys.version_info.major == 2:
        dictionary = pickle.load(f)
    elif sys.version_info.major == 3:
        dictionary = pickle.load(f,encoding='latin-1')
    id2wd = {v: i for i, v in dictionary.items()}

model = Seq2seq(1, len(dictionary), args.unit)
serializers.load_npz(args.modelPath, model)

parser = lambda x: MeCab.Tagger("-Owakati -d /lib/mecab-ipadic-neologd").parse(x).split(" ")
def main():
    '''
    # Conversation part wait user input and return it.
    while True:
        try:
            sys.stdout.write("David:")
            utterance = input()
            if utterance == "exit":
                print("じゃあね!!")
                sys.exit(0)

            utterLine = parser(zh.z2h(utterance).lower())
            utterLineR = utterLine[::-1]
            conversation(utterLineR, model, dictionary, id2wd)
        except KeyError:
            print("う〜ん、おにぃちゃんっ！よくわかんないから別の言葉で言ってよね！もうっ :/")
    '''

if __name__ == '__main__':
    print("ぽよ")
