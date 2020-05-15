import mxnet as mx
import numpy as np
from mxnet import gluon, nd, autograd
from mxnet.gluon import Block, nn
import gluonnlp as nlp
import gluonnlp.model.transformer as trans
from random import choice
from opencc import OpenCC
from zhon import cedict, hanzi, zhuyin
import string
from utils import load_pickle

from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as sce
from pinyin import PinYinSampler

s2t = OpenCC('s2t')  # 
t22 = OpenCC('t2s')
fullwidth_digit = ''.join(['１','２','３','４','５','６','７','８','９','１０'])

counter_tgt = nlp.data.count_tokens('． ' + fullwidth_digit + cedict.traditional + hanzi.punctuation + zhuyin.characters + zhuyin.marks + string.printable)
vocab_tgt = nlp.vocab.BERTVocab(counter_tgt)

extra_vocab = load_pickle('tencent_vocab_tc')

options = dict()

options['char_error_rate'] = 0.5
options['word_error_rate'] = 0.5

PS = PinYinSampler(list(vocab_tgt.token_to_idx.keys()), extra_vocab, options)

target_text = "他是一個很奇怪的人，每天都躲在家裡"

PS.errorize_sentence(target_text)