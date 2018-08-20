import os
import sys
import copy
import time
import math
import random

import numpy as np
import functools as ft

import chainer
from chainer import cuda
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

yaju = 810

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# for decoder
class NLayerLSTM(chainer.ChainList):
    def __init__(self, eDim, hDim):
        layers = [0] * 1  # get place holder for each layer
        layers[0] = L.LSTM(eDim, hDim)
        super(NLayerLSTM, self).__init__(*layers)

    # process a lstm for every layer
    def __call__(self, hin):
        hout = self[0](hin)
        return hout

    # initialize all layer's lstm's state
    def reset_state(self):
         self[0].reset_state()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # for passing cell and hidden state from encoder to decoder and
    # beam search where multiple lstm apply is not able

    # [0c, 0h,
    #  1c, 1h,
    #  2c, 2h,
    #  ...
    #  nc, nh]

    # ic, ih: np.array
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    def getAllLSTMStates(self):
        h = self[0].h
        c = self[0].c
        return (h, c)

    def setAllLSTMStates(self, states): # states: (h, c)
        self[0].h = states[0]
        self[0].c = states[1]


class EncoderDecoder:
    def __init__(self, encoderVocab, decoderVocab, trainData, setting):
        self.encw2i = encoderVocab # dict
        self.decw2i = decoderVocab # dict
        self.encVocabSize = len(encoderVocab)
        self.decVocabSize = len(decoderVocab)

        # dictionary for get vocabulary from index
        self.enci2w  = {v: k for k, v in self.encw2i.items()}
        self.deci2w = {v: k for k, v in self.decw2i.items()}

        self.eDim_enc = setting.eDim_enc
        self.eDim_dec = setting.eDim_dec

        self.hDim = setting.hDim
        self.n_layers = setting.n_layers

        self.dropout_rate = setting.dropout_rate
        self.gpu = setting.gpu

    # encoder-docoder network
    def initModel(self):
        self.model = chainer.Chain(
            encoderEmbed=L.EmbedID(self.encVocabSize, self.eDim_enc), # encoder embedding layer
            decoderEmbed=L.EmbedID(self.decVocabSize, self.eDim_dec), # decoder embedding layer
            decOutputL=L.Linear(self.hDim, self.decVocabSize), # output layer
            encoder_bak=L.NStepLSTM(n_layers=self.n_layers, in_size=self.eDim_enc, out_size=self.hDim, dropout=self.dropout_rate), # encoder backward
            encoder_fwd=L.NStepLSTM(n_layers=self.n_layers, in_size=self.eDim_enc, out_size=self.hDim, dropout=self.dropout_rate), # encoder forward
            decoder_=NLayerLSTM(eDim=self.eDim_dec, hDim=self.hDim), # decoder
            attnIn=L.Linear(self.hDim, self.hDim, nobias=True), # attn
            attnOut=L.Linear(self.hDim + self.hDim, self.hDim, nobias=True) # attn
        )

    def getLastSavedIndex(self, dirname):
        for dirpath, dirnames, filenames in os.walk(dirname): # model_0.npz
            norm_filenames = [fn for fn in filenames if not(fn == ".DS_Store")]
            ansind = 0 if (len(norm_filenames) == 0) else max([int(fn.split(".")[0].split("_")[1])
                                                          for fn in list(norm_filenames)])
        return ansind

    def saveModel(self, dirname, diffepoch):
        ind = self.getLastSavedIndex(dirname)
        fname = dirname + "model_{}".format(int(ind) + int(diffepoch))

        copied_model = copy.deepcopy(self.model)
        copied_model.to_cpu()

        sys.stdout.write("# Saved model: {}\n".format(fname))
        serializers.save_npz(fname, copied_model)

    def loadModel(self, dirname):
        ind = self.getLastSavedIndex(dirname)
        if ind > 0:
            fname = dirname + "model_{}".format(int(ind))
            sys.stdout.write("# Loaded model: {}\n".format(fname))
            serializers.load_npz(fname, self.model)
        else:
            sys.stdout.write("# No model loaded\n")

    def setToGPUs(self):
        if self.gpu >= 0:
            sys.stderr.write("# Working on GPU [gpu=%d]\n" % (self.gpu))

            self.model.encoderEmbed.to_gpu(self.gpu)
            self.model.decoderEmbed.to_gpu(self.gpu)

            self.model.decOutputL.to_gpu(self.gpu)

            self.model.encoder_bak.to_gpu(self.gpu)
            self.model.encoder_fwd.to_gpu(self.gpu)
            self.model.decoder_.to_gpu(self.gpu)

            self.model.attnIn.to_gpu(self.gpu)
            self.model.attnOut.to_gpu(self.gpu)

        else:
            sys.stderr.write("# Working on CPU [cpu=%d]\n" % (self.gpu))

    # get embedding for encoder
    def getEncoderInputEmbeddings(self, xs): # xs: [arr(l1), ...], still on cpu
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        vxs = [F.copy(chainer.Variable(x), self.gpu) for x in xs] # to gpu
        ex = self.model.encoderEmbed(F.concat(tuple(vxs), axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return list(exs)

    # get embedding for decoder
    def getDecoderInputEmbeddings(self, xs): # xs: [arr(l1), ...], still on cpu
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        vxs = [F.copy(chainer.Variable(x), self.gpu) for x in xs] # to gpu
        ex = self.model.decoderEmbed(F.concat(tuple(vxs), axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return list(exs)

    def outputMerging(self, bak, fwd): # bak, fwd: [(encLen, hDim)]
        ys = [b + f for b, f in zip(bak, fwd)]
        return ys

    def statesMerging(self, sb, sf): # sb, sf: (layer, batch, hDim)
        s_added = sb + sf;
        s = F.stack([s_added[-1]]) # use last encoder's layer's hidden state for decoder's state
        return s

    def calcAttention(self, h1, encList, encLen, batchsize): # attention, h1: (batch, hDim)
        target1 = self.model.attnIn(h1)  # convert
        # (batchsize, self.hDim) => (batchsize, self.hDim)
        target2 = F.expand_dims(target1, axis=1)
        # (batchsize, self.hDim) => (batchsize, 1, self.hDim)
        target3 = F.broadcast_to(target2, (batchsize, encLen, self.hDim))
        # (batchsize, 1, self.hDim) => (batchsize, encLen, self.hDim)

        # bilinear
        # target3: (batchsize, encLen, self.hDim) tensor
        # encList: (batchsize, encLen, self.hDim) tensor

        # [[[...], [...], [...], ...]]
        #     *      *      *
        # [[[...], [...], [...], ...]]
        # [[ a0,    a1,    a2, ...]]

        aval = F.sum(target3 * encList, axis=2) # shape: (batchsize, encLen)

        """
        # MLP
        # convert for attnSum
        t1 = F.reshape(target3, (batchsize * encLen, self.hDim))
        # (batchsize * encLen, self.hDim) => (batchsize * encLen, 1)
        t2 = self.model.attnSum(F.tanh(t1 + aList))
        # shape: (batchsize, encLen)
        aval = F.reshape(t2, (batchsize, encLen))
        """

        # 3, calc softmax
        cAttn1 = F.softmax(aval) # (batchsize, encLen) => (batchsize, encLen)

        # 4, make context vector using attention
        cAttn2 = F.expand_dims(cAttn1, axis=1) # (batchsize, encLen) => (batchsize, 1, encLen)

        cAttn3 = F.batch_matmul(cAttn2, encList)
        # (1, encLen) x (encLen, hDim) matmul for batchsize times => (batchsize, 1, hDim)

        context = F.reshape(cAttn3, (batchsize, self.hDim)) # (batchsize, hDim)

        # 6, attention時の最終隠れ層の計算
        c1 = F.concat((h1, context))
        c2 = self.model.attnOut(c1)
        finalH = F.tanh(c2)

        return finalH  # context

    def encodingInput(self, encarrs):
        xs_bak = [x[::-1] for x in encarrs] # for backward
        xs_fwd = [x for x in encarrs] # for forward

        exs_bak = self.getEncoderInputEmbeddings(xs_bak) # for backward
        exs_fwd = self.getEncoderInputEmbeddings(xs_fwd) # for forward

        hx_bak, cx_bak, xs_bak = self.model.encoder_bak(None, None, exs_bak) # for backward
        hx_fwd, cx_fwd, xs_fwd = self.model.encoder_fwd(None, None, exs_fwd) # for forward

        xs = self.outputMerging(xs_bak, xs_fwd)

        hx = self.statesMerging(hx_bak, hx_fwd) # (1, batch, hDim)
        cx = self.statesMerging(cx_bak, cx_fwd) # (1, batch, hDim)

        return hx, cx, xs

    def train(self, optimizer, trainData, epoch):
        xp = cuda.get_array_module(self.model.decOutputL.W.data)

        total_loss_val = 0

        for enu, (enc, dec) in enumerate(trainData):
            begin = time.time()

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            hy, cy, ys = self.encodingInput(enc) # encoding
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # data preparing and embedding
            dec_ind_dom = [y[:-1] for y in dec] # for pred
            dec_ind_cod = [F.copy(chainer.Variable(y[1:]), self.gpu) for y in dec] # for true
            cword = sum([len(y) for y in dec_ind_cod])
            decoder_dom = self.getDecoderInputEmbeddings(dec_ind_dom)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # decoding
            self.model.decoder_.reset_state()
            self.model.decoder_.setAllLSTMStates((hy[0], cy[0]))

            zs_ = [] # [(batch, hDim)]
            decoder_dom_ = F.swapaxes(F.stack(decoder_dom), 0, 1) # (decLen, batch, hDim)
            dec_ind_cod_ = F.swapaxes(F.stack(dec_ind_cod), 0, 1) # (decLen, batch)

            for i in range(len(decoder_dom_)):
                hout_ = self.model.decoder_(decoder_dom_[i])
                hout_att = self.calcAttention(h1=hout_, encList=F.stack(ys), encLen=len(ys[0]), batchsize=len(ys))
                zs_.append(hout_att)

            concat_zs_pred_ = F.concat(tuple(zs_), axis=0)
            concat_zs_true_ = F.concat(tuple(dec_ind_cod_), axis=0)
            closs_ = F.sum(F.softmax_cross_entropy(self.model.decOutputL(concat_zs_pred_),
                                                 concat_zs_true_,
                                                 reduce="no")) / cword
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

            if (xp.isnan(closs_.data)):
                sys.stderr.write("Nan occured! skip :(\n")
                continue

            total_loss_val += closs_.data

            optimizer.target.cleargrads()
            closs_.backward()
            optimizer.update()

            end = time.time()
            sys.stdout.write("epoch: {:3d}, enu: {:3d}, X-entropy: {:5.6f}, total_word: {:5d}, time: {:2.5f}\n".format(
                             int(epoch), int(enu), float(closs_.data), int(cword), float(end - begin)))

            if enu == 1200: break

        next_test_index = random.randint(0, len(trainData) - 1)
        arr_list = trainData[next_test_index][0]

        for arr in arr_list[:5]:
            fuck = self.translate([arr])
            sys.stdout.write("# Test: {}, utter: [{}], rep: [{}]\n".format(
                             int(epoch), "".join([self.enci2w[x] for x in arr]), fuck[0][3:-4]))

        sys.stderr.write("total_loss: {:4.5f}\n".format(float(total_loss_val)))

    def translate(self, encarrs, max_length=10, beam_width=5): # len(xs) = 1
        xp = cuda.get_array_module(self.model.decOutputL.W.data)

        assert (len(encarrs) == 1), "Batch size is too match!"

        with chainer.no_backprop_mode(), chainer.using_config("train", False):
            hx, cx, xs = self.encodingInput(encarrs)

            # [(log (p1 * p2 * ...), (h, c), [word1, worc2, ...])]
            # result = [(0, (hx, cx), ["<s>"])] # priority queue for beam search
            # [(log (p1 * p2 * ...), ((1, batch, hDim), (batch, hDim)), [word1, worc2, ...])]

            result_ = [(0.0, (hx[0], cx[0]), ["<s>"])] # priority queue for beam search
            # [(log (p1 * p2 * ...), ((batch, hDim), (batch, hDim)), [word1, worc2, ...])]

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            i_ = 0

            #beam search
            while True:
                c_result_ = []
                for (logp, (h, c), stkwords) in result_:
                    lastword_ = stkwords[-1]
                    y_ = xp.full(1, self.decw2i[lastword_], 'i')
                    ey_ = self.model.decoderEmbed(y_)

                    self.model.decoder_.reset_state()
                    self.model.decoder_.setAllLSTMStates((h, c))

                    dy_ = self.model.decoder_(ey_)
                    dy_att = self.calcAttention(h1=dy_, encList=F.stack(xs), encLen=len(xs[0]), batchsize=len(xs))
                    (nh_, nc_) = self.model.decoder_.getAllLSTMStates()

                    wy_ = F.softmax(self.model.decOutputL(dy_att))
                    yind_ = xp.argmax(wy_.data, axis=1).astype('i')

                    probs_list_ = cuda.to_cpu(wy_.data)[0].tolist()
                    index_prob_list_ = list(zip(list(range(len(probs_list_))), probs_list_))

                    for idx, prob in index_prob_list_:
                        if self.deci2w[idx] == "<\s>":
                            c_result_.append((logp + 0,
                                            (nh_, nc_),
                                            stkwords + [self.deci2w[idx]]))
                            sys.stdout.write("{}\n", c_result_)
                        else:
                            c_result_.append((logp + math.log(prob + 1e-100),
                                            (nh_, nc_),
                                            stkwords + [self.deci2w[idx]]))

                sorted_c_result_ = sorted(c_result_, key=lambda x: x[0], reverse=True)
                result_ = sorted_c_result_[:beam_width]

                end = ft.reduce(lambda a, b: a or b,
                                ["<\s>" in x[2] for x in result_]) # check output "</s>" or not

                i_ += 1
                if end or i_ == max_length: break

            outs_ = [x[2] for x in result_]
            outs_ = ["".join(x).split("</s>")[0] + "</s>" if ("</s>" in x) else "".join(x) + "</s>" for x in outs_]
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        return outs_
