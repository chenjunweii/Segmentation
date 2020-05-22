from zhon import hanzi, zhuyin, cedict
import string
from utils import load_pickle
import os
import jaconv
import sys
fullwidth_digit = ''.join(['１','２','３','４','５','６','７','８','９','１０'])


"""
let inputs = document.getElementsByTagName("input")
for (let v of inputs){
     v.checked = true   
}
"""

config = dict()

config['dir'] = '/home/ur/sent_seg'
config['int_max_length'] = 128 # dataset 目前最長也是 256
config['list_puncuation_marks'] = [s for s in hanzi.punctuation + zhuyin.marks + string.punctuation]
config['str_character_target'] = '． ' + fullwidth_digit + cedict.traditional + hanzi.punctuation + zhuyin.characters + zhuyin.marks + string.printable + jaconv.h2z(string.ascii_letters, ascii = True)
config['float_pm_random_remove_rate'] = 0.25
config['float_pm_random_add_rate'] = 0.25 # 隨機將 pm 換成其他 pm 的機率
config['float_pm_random_error_rate'] = 0.25 # 隨機將在不是 pm 的 char 前面加上隨機的 pm
config['float_pm_random_bypass_rate'] = 0.25 # 隨機將在不是 pm 的 char 前面加上隨機的 pm
config['float_char_error_rate'] = 0.05 # 
config['float_word_error_rate'] = 0.75
config['csc_fixed'] = True
config['arch_name'] = 'bert_128_with_constraint_no_pretrain'
config['int_val_set'] = 1000
config['val_freq'] = 2500
config['use_encoder_constraint'] = True

# print(os.path.join(config['dir'], 'embedding/tencent_vocab_tc' ))
config['list_extra_words'] = load_pickle(os.path.join(config['dir'], 'embedding/tencent_vocab_tc'))


config['train'] = dict(); config['test'] = dict()

config['train']['batch_size'] = 16
config['test']['batch_size'] = 1


config['dataset'] = dict()
config['dataset']['test'] = ["1080315"]
config['dataset']['train'] = ["1081015", "1071015", "1051031", "1060315", "1061031", "1070315", "1050315", "1040315", "1041031"]


