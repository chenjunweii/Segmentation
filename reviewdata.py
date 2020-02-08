from mxnet.gluon.data import SimpleDataset, DataLoader
from mxnet import nd
import numpy as np
from copy import deepcopy
from utils import load_pickle
import re

class ReviewData(object):
  def __init__(self, tokenizer, transformer, batch_size, pms, actions):
    self.pms = pms
    self.tokenizer = tokenizer
    self.transformer = transformer
    
    self.batch_size = batch_size
    
    self.actions = actions
    self.map_pms_idx = { pm : idx for idx, pm in enumerate(self.pms) }
    self.map_actions_idx = { action : idx for idx, action in enumerate(self.actions) }
    
    self.data = self._load_cache()
    
    
    # self.train_data = [1, 2, 3, 4, 5]
    
  def _load_cache(self):
  
    return load_pickle('reviews.cache')
  
    
  def _build_target_embedding(self, str_target_text, str_input_text, list_replaced_pms, list_replaced_idxs):
  
    # target_action 和 target_text 的長度會不同， target_action + input_text => target_text
  
    list_target_actions = [None] * (512)#len(str_input_text) + 1) # +1 是爲了多預測一格讓我們可以在頭尾都做預測
    
    list_target_pms = [None] * 512#(len(str_input_text) + 1)
    
    for _idx, _pm in zip(list_replaced_idxs, list_replaced_pms):
    
      list_target_pms[_idx + 1] = _pm
      
      list_target_actions[_idx + 1] = 'add'
      
    list_target_actions[0] = '[START]'
    
    list_target_pms[0] = '[START]'
      
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
  
  def _erroize_pm(self, text):
  
    pass
  
  def get_loader(self):
  
    def batchify_fn(list_target_texts):
    
      words = []; valid_lens = []; segments = []; target_actions = []; target_pms = []; list_input_texts = []
      
      _list_target_texts = []
      
      for str_target_text in list_target_texts:
      
        if len(str_target_text) > 100 or len(str_target_text) < 10:
        
          continue
          
        _list_target_texts.append(str_target_text)
      
        str_input_text, list_replaced_pm, list_replaced_idx = self._remove_pm(str_target_text)
        
        target_action, target_pm = self._build_target_embedding(str_target_text, str_input_text, list_replaced_pm, list_replaced_idx) 
      
        sample = self.transformer([str_input_text])
        
        word, valid_len, segment = nd.array([sample[0]]), nd.array([sample[1]]), nd.array([sample[2]])
        
        words.append(word.astype(np.float32)); valid_lens.append(valid_len.astype(np.float32)); segments.append(segment.astype(np.float32))
        
        target_actions.append(target_action.astype(np.float32)); target_pms.append(target_pm.astype(np.float32));
        
        list_input_texts.append(str_input_text)
      
      return nd.concat(*words, dim = 0), nd.concat(*valid_lens, dim = 0), nd.concat(*segments, dim = 0), nd.concat(*target_actions, dim = 0), nd.concat(*target_pms, dim = 0), list_input_texts, _list_target_texts
  
    self.dataset = SimpleDataset(self.data)
    
    self.loader = DataLoader(self.dataset, batch_size = self.batch_size, batchify_fn = batchify_fn)
    
    return self.loader
    
  
    
    
    