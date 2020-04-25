from zhon import hanzi, zhuyin, cedict
import string
from utils import load_pickle

fullwidth_digit = ''.join(['１','２','３','４','５','６','７','８','９','１０'])


"""
let inputs = document.getElementsByTagName("input")
for (let v of inputs){
     v.checked = true   
}
"""

config = dict()

config['int_max_length'] = 256 # dataset 目前最長也是 256
config['list_puncuation_marks'] = [s for s in hanzi.punctuation + zhuyin.marks + string.punctuation]
config['str_character_target'] = '． ' + fullwidth_digit + cedict.traditional + hanzi.punctuation + zhuyin.characters + zhuyin.marks + string.printable
config['float_pm_remove_rate'] = 1
config['float_pm_random_rate'] = 0.0 # 隨機將 pm 換成其他 pm 的機率
config['float_pm_random_location_rate'] = 0.0 # 隨機將在不是 pm 的 char 前面加上隨機的 pm
config['float_char_error_rate'] = 0.10 # 
config['float_word_error_rate'] = 0.75

config['arch_name'] = 'bert_cged16_fgx1_lamb_decoder_6'

config['list_extra_words'] = load_pickle('tencent_vocab_tc')

config['train'] = dict(); config['test'] = dict()

config['train']['batch_size'] = 16
config['test']['batch_size'] = 1


config['dataset'] = dict()
config['dataset']['test'] = ["1080315"]
config['dataset']['train'] = ["1081015", "1071015", "1051031", "1060315", "1061031", "1070315", "1050315", "1040315", "1041031"]


