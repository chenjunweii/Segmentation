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
    
    print('mode {} => {}'.format(self.mode, len(self.data)))
    
    
    
    # self.train_data = [1, 2, 3, 4, 5]
    
  def _load_cache(self):
  
    return load_pickle('cache/reviews.cache')
    
  def convert_actions_to_indexs(self, tokens):
  
    return [self.map_actions_idx[t] for t in tokens]
    
  def convert_pms_to_indexs(self, tokens):
  
    return [self.map_pms_idx[t] for t in tokens]
    
  def _build_target_pm(self, ):
  
    pass
    
  def _build_target_action(self,):
  
    pass
    
  def _remove_pm(self, target_text):
    
    # list_replaced_idxs = []
    # list_replaced_pms = []
    input_text = ''
    for _idx, _target_text in enumerate(target_text):
      if _target_text in self.pms and np.random.ranf() < self.config['float_pm_remove_rate']:
        # list_replaced_idxs.append(len(input_text))
        # list_replaced_pms.append(_target_text)
        pass
      elif _target_text in self.pms and np.random.ranf() < self.config['float_pm_random_rate']:
        pms_random = deepcopy(self.pms)
        pms_random.remove(_target_text)
        pm_random = choice(pms_random)
        input_text = input_text + pm_random
      elif _target_text not in self.pms and np.random.ranf() < self.config['float_pm_random_location_rate']:
        pm_random = choice(self.pms)
        input_text = input_text + pm_random + _target_text
      else:
        input_text = input_text + _target_text
        
    return input_text

  def _swap_word_order(self, target_text):
  
    pass
  
  def _add_redundancy(self, input_text):
    pass
    
  def _remove_char(self, input_text):
  
    pass
  
  def _transform_target(self, target_text):
  
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
      
      _list_target_texts = []
      
      if self.mode == 'train': # 先暫時這樣本來 test 應該走 else
        for str_target_text in list_target_texts:
        
          # if np.random.ranf() > 0.5:
          #   str_input_text = self.structure.randomize_word_order(str_target_text)
          # else:
          str_input_text = self.pinyin_sampler.errorize_sentence(str_target_text)
          str_input_text = self._remove_pm(str_input_text)
          input_data = self.transformer([str_input_text])
          target_data = self._transform_target(str_target_text)
          
          if len(target_data[0]) > self.max_seq_len or input_data[0].shape[0] > self.max_seq_len: # 超過長度
            continue
    
          input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])
          target_word, target_valid_len = nd.array([target_data[0]]), nd.array([target_data[1]])
          target_segment = input_segment
          
          _list_target_texts.append(str_target_text)
          input_words.append(input_word.astype(np.float32)); input_valid_lens.append(input_valid_len.astype(np.float32)); input_segments.append(input_segment.astype(np.float32))
          target_words.append(target_word.astype(np.float32)); target_valid_lens.append(target_valid_len.astype(np.float32));
          target_segments.append(target_segment.astype(np.float32))
          # target_actions.append(target_action.astype(np.float32)); target_pms.append(target_pm.astype(np.float32));
          
          list_input_texts.append(str_input_text)
        
        return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0), list_input_texts, _list_target_texts
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
    self.loader = DataLoader(self.dataset, batch_size = self.batch_size, batchify_fn = batchify_fn, shuffle = False, last_batch = 'rollover')
    
    return self.loader
    
  
    
    
    