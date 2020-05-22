from mxnet.gluon.data import SimpleDataset, DataLoader
from mxnet import nd
import numpy as np
from copy import deepcopy
from random import choice
from utils import load_pickle
import jieba
import re
from pinyin import PinYinSampler
from opencc import OpenCC
from structure import Structure
s2t = OpenCC('s2t')  # 
t2s = OpenCC('t2s') 
class ReviewData(object):
  def __init__(self, tokenizer, transformer, vocab_tgt, config, mode):
    self.pms = config['list_puncuation_marks']
    self.tokenizer = tokenizer
    self.transformer = transformer
    self.mode = mode
    self.batch_size = config[mode]['batch_size']
    self.max_seq_len = config['int_max_length']
    self.vocab_tgt = vocab_tgt
    self.config = config
    self.extra_vocab = config['list_extra_words']
    self.pinyin_sampler = PinYinSampler(list(vocab_tgt.token_to_idx.keys()), self.extra_vocab, self.config)
    # self.actions = actions
    self.map_pms_idx = { pm : idx for idx, pm in enumerate(self.pms) }
    # self.map_actions_idx = { action : idx for idx, action in enumerate(self.actions) }
    self.structure = Structure()
    self.data = self._load_cache()[mode]
    
  def _load_cache(self):
  
    return load_pickle('cache/reviews.cache')
    
  def convert_actions_to_indexs(self, tokens):
  
    return [self.map_actions_idx[t] for t in tokens]
    
  def convert_pms_to_indexs(self, tokens):
  
    return [self.map_pms_idx[t] for t in tokens]
    
  def errorize_pm(self, target_text):
    input_text = ''
    pm_remove_idx = []
    pm_add_idx = []
    pm_error_idx = []
    
    action = np.random.choice(['remove', 'error', 'add', 'bypass'], p = [self.config['float_pm_random_remove_rate'], self.config['float_pm_random_error_rate'], self.config['float_pm_random_add_rate'], self.config['float_pm_random_bypass_rate']] )
    
    for _idx, _target_text in enumerate(target_text):
      if _target_text in self.pms and action == 'remove':
        __idx = len(input_text)
        if __idx < self.max_seq_len:
          pm_remove_idx.append(__idx)
      elif _target_text in self.pms and action == 'error':
        pms_random = deepcopy(self.pms)
        pms_random.remove(_target_text)
        pm_random = choice(pms_random) # 選除了自己以外的
        input_text = input_text + pm_random
        __idx = len(input_text)
        if __idx < self.max_seq_len:
          pm_error_idx.append(__idx)
      elif _target_text not in self.pms and action == 'add':
        pm_random = choice(self.pms)
        input_text = input_text + pm_random + _target_text
        __idx = len(input_text) - 1
        if len(input_text) < self.max_seq_len:
          pm_add_idx.append(__idx)
        # error_idx.append(len(input_text))
      else:
        input_text = input_text + _target_text
    return input_text, pm_error_idx, pm_add_idx, pm_remove_idx
  
  def transform_pm_error(self, pm_error_idx, pm_add_idx, pm_remove_idx):
  
    pm_error_emb = np.zeros([1, self.max_seq_len, 1])
    pm_add_emb = np.zeros([1, self.max_seq_len, 1]) 
    pm_remove_emb = np.zeros([1, self.max_seq_len, 1]) 
    
    for _idx in pm_add_idx:
      pm_add_emb[0, _idx] = 1
    for _idx in pm_error_idx:
      pm_error_emb[0, _idx] = 1
    for _idx in pm_remove_idx:
      pm_remove_emb[0, _idx] = 1
    return nd.array(pm_error_emb), nd.array(pm_add_emb), nd.array(pm_remove_emb)
  
  def transform_target(self, target_text):
    target_text = [self.vocab_tgt.cls_token] + [_text for _text in target_text] + [self.vocab_tgt.sep_token]
    valid_len = len(target_text)
    target_text = target_text + [self.vocab_tgt.padding_token] * (self.max_seq_len - valid_len)
    idx = [self.vocab_tgt.token_to_idx[_text] for _text in target_text]
    
    # if 0 in idx:
    
    #   _idx = idx.index(0)
      
    #   print(target_text)
      
    #   print(self.vocab_tgt.token_to_idx['[UNK]'])
      
    #   print(target_text[_idx])
    
    #   raise
    
    # print(self.vocab_tgt)
    
    # print(target_text)
    
    # print(idx)
    
    # raise
    # return np.expand_dims(np.array(idx), 0), np.expand_dims(np.array(valid_len), 0), # segment => same as input
    return idx, valid_len # segment => same as input

  
  
    
  
  def get_loader(self):
  
    def batchify_fn(list_target_texts):
    
      input_words = []; input_valid_lens = []; input_segments = []
      target_words = []; target_valid_lens = []; target_segments = []
      target_actions = []; target_pms = []; list_input_texts = []
      _list_target_texts = []; pm_error_idxs = []; pm_add_idxs = []; pm_remove_idxs = []
      
      if self.mode == 'train': # 先暫時這樣本來 test 應該走 else
      
        for str_target_text in list_target_texts:
        
          # if np.random.ranf() > 0.5:
          #   str_input_text = self.structure.randomize_word_order(str_target_text)
          # else:
          # str_input_text = self.pinyin_sampler.errorize_sentence(str_target_text)
          str_input_text, pm_error_idx, pm_add_idx, pm_remove_idx = self.errorize_pm(str_target_text)
          input_data = self.transformer([str_input_text])
          target_data = self.transform_target(str_target_text)
          if len(target_data[0]) > self.max_seq_len or input_data[0].shape[0] > self.max_seq_len: # 超過長度
            continue
          pm_error_idx, pm_add_idx, pm_remove_idx = self.transform_pm_error(pm_error_idx, pm_add_idx, pm_remove_idx)
          
          input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])
          target_word, target_valid_len = nd.array([target_data[0]]), nd.array([target_data[1]])
          target_segment = input_segment
          
          _list_target_texts.append(str_target_text)
          input_words.append(input_word.astype(np.float32)); input_valid_lens.append(input_valid_len.astype(np.float32)); input_segments.append(input_segment.astype(np.float32))
          target_words.append(target_word.astype(np.float32)); target_valid_lens.append(target_valid_len.astype(np.float32));
          target_segments.append(target_segment.astype(np.float32))
          pm_add_idxs.append(pm_add_idx)
          pm_error_idxs.append(pm_error_idx)
          pm_remove_idxs.append(pm_remove_idx)
          # target_actions.append(target_action.astype(np.float32)); target_pms.append(target_pm.astype(np.float32));
          
          list_input_texts.append(str_input_text)
          
        return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0), nd.concat(*pm_error_idxs, dim = 0), nd.concat(*pm_add_idxs, dim = 0), nd.concat(*pm_remove_idxs, dim = 0), list_input_texts, _list_target_texts
      # return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0)#, nd.concat(*target_actions, dim = 0), nd.concat(*target_pms, dim = 0), list_input_texts, list_target_texts
      
      
      else:
      
        # print(list_target_texts)
        # print(len(list_target_texts))
        assert(len(list_target_texts) == 1)
        # for test_pair in list_target_texts:
        str_input_text = list_target_texts[0][0]
        str_target_text = list_target_texts[0][1]
        return str_input_text, str_target_text
          
    self.dataset = SimpleDataset(self.data)
    shuffle = True if self.mode == 'train' else False 
    self.loader = DataLoader(self.dataset, batch_size = self.batch_size, batchify_fn = batchify_fn, shuffle = shuffle, last_batch = 'rollover')
    
    return self.loader
    
  
    
    
    