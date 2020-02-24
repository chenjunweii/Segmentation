from zhon import hanzi, zhuyin, cedict
import string
from utils import load_pickle

fullwidth_digit = ''.join(['１','２','３','４','５','６','７','８','９','１０'])


config = dict()

config['int_max_length'] = 256 # dataset 目前最長也是 256
config['list_puncuation_marks'] = [s for s in hanzi.punctuation + zhuyin.marks + string.punctuation]
config['str_character_target'] = '． ' +fullwidth_digit + cedict.traditional + hanzi.punctuation + zhuyin.characters + zhuyin.marks + string.printable
config['float_pm_remove_rate'] = 0.7
config['float_char_error_rate'] = 0.5
config['float_word_error_rate'] = 0.5

config['list_extra_words'] = load_pickle('tencent_vocab_tc')

config['train'] = dict(); config['test'] = dict()

config['train']['batch_size'] = 16
config['test']['batch_size'] = 32

config['dataset'] = dict()
config['dataset']['test'] = ["1080315"]
config['dataset']['train'] = ["1081015", "1071015"]
