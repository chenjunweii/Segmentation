from mxnet.gluon.data import SimpleDataset, DataLoader
from mxnet import nd
import numpy as np
from copy import deepcopy
from utils import load_pickle
import re
from opencc import OpenCC                
s2t = OpenCC('s2t')  # 
t2s = OpenCC('t2s') 
class ReviewData(object):
  def __init__(self, tokenizer, transformer, vocab_tgt, batch_size, pms, max_seq_len):
    self.pms = pms
    self.tokenizer = tokenizer
    self.transformer = transformer
    
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.vocab_tgt = vocab_tgt
    
    # self.actions = actions
    self.map_pms_idx = { pm : idx for idx, pm in enumerate(self.pms) }
    # self.map_actions_idx = { action : idx for idx, action in enumerate(self.actions) }
    
    self.data = self._load_cache()
    
    
    
    # self.train_data = [1, 2, 3, 4, 5]
    
  def _load_cache(self):
  
    return load_pickle('reviews.cache')
  
  def _build_target_embedding(self, str_target_text, str_input_text):
  
    print(self.transformer(str_input_text))
    
    raise
    
  def dd_build_target_embedding(self, str_target_text, str_input_text, list_replaced_pms, list_replaced_idxs):
  
    # target_action 和 target_text 的長度會不同， target_action + input_text => target_text
  
    list_target_actions = [None] * (512)#len(str_input_text) + 1) # +1 是爲了多預測一格讓我們可以在頭尾都做預測
    
    list_target_pms = [None] * 512#(len(str_input_text) + 1)
    
    for _idx, _pm in zip(list_replaced_idxs, list_replaced_pms):
    
      list_target_pms[_idx] = _pm
      
      list_target_actions[_idx] = 'add'
      
    # list_target_actions[0] = '[START]'
    
    # list_target_pms[0] = '[START]'
      
    return nd.array(self.convert_actions_to_indexs(list_target_actions)).expand_dims(0), nd.array(self.convert_pms_to_indexs(list_target_pms)).expand_dims(0)
    
  def convert_actions_to_indexs(self, tokens):
  
    return [self.map_actions_idx[t] for t in tokens]
    
  def convert_pms_to_indexs(self, tokens):
  
    return [self.map_pms_idx[t] for t in tokens]
    
  def _build_target_pm(self, ):
  
    pass
    
  def _build_target_action(self,):
  
    pass
    
  def _remove_pm(self, target_text):
    
    list_replaced_idxs = []
    list_replaced_pms = []
    input_text = ''
    for _idx, _target_text in enumerate(target_text):
      if _target_text in self.pms:
        list_replaced_idxs.append(len(input_text))
        list_replaced_pms.append(_target_text)
      else:
        input_text = input_text + _target_text
        
    # print('Target : ', target_text)
    # print('Input : ', input_text)
    return input_text, list_replaced_pms, list_replaced_idxs
    
  def _remove_modify(self, target_text):
    return
    
  
  def _transform_target(self, target_text):
  
    target_text = [self.vocab_tgt.cls_token] + [_text for _text in target_text] + [self.vocab_tgt.sep_token]
    
    valid_len = len(target_text)
    
    target_text = target_text + [self.vocab_tgt.padding_token] * (self.max_seq_len - valid_len)
    
    idx =  [self.vocab_tgt.token_to_idx[_text] for _text in target_text]
    
    # print(self.vocab_tgt)
    
    # print(target_text)
    
    # print(idx)
    
    # raise
    # return np.expand_dims(np.array(idx), 0), np.expand_dims(np.array(valid_len), 0), # segment => same as input
    return idx, valid_len # segment => same as input

  
  def _erroize_pm(self, text):
  
    pass
  
  def get_loader(self):
  
    def batchify_fn(list_target_texts):
    
      input_words = []; input_valid_lens = []; input_segments = []
      
      target_words = []; target_valid_lens = []; target_segments = []
      
      target_actions = []; target_pms = []; list_input_texts = []
      
      _list_target_texts = []
      
      for str_target_text in list_target_texts:
        
        str_input_text, list_replaced_pm, list_replaced_idx = self._remove_pm(str_target_text)
  
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
      
      # return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0)#, nd.concat(*target_actions, dim = 0), nd.concat(*target_pms, dim = 0), list_input_texts, list_target_texts
      return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0), list_input_texts, _list_target_texts
    self.dataset = SimpleDataset(self.data)
    self.loader = DataLoader(self.dataset, batch_size = self.batch_size, batchify_fn = batchify_fn, shuffle = True, last_batch = 'rollover')
    
    return self.loader
    
  
    
    
    