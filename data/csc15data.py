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
from nltk.translate.bleu_score import sentence_bleu
s2t = OpenCC('s2t')  # 
t2s = OpenCC('t2s') 
class SighanCSC15Data(object):
  def __init__(self, tokenizer, transformer, config, mode, vocab_tgt = None, useDecoder = False, args = None):
    self.tokenizer = tokenizer
    self.transformer = transformer
    self.mode = mode
    self.batch_size = config[mode]['batch_size']
    self.max_seq_len = config['int_max_length']
    self.vocab_tgt = vocab_tgt
    self.config = config
    self.extra_vocab = config['list_extra_words']
    self.compare_counter = 0
    self.compare_counter_same = 0
    self.compare_counter_diff = 0
    self.bleu_scores = []
      
    self.data = self._load_cache()[mode]
  def _load_cache(self):
    return load_pickle('cache/sighancsc15.cache')
    
  def convert_actions_to_indexs(self, tokens):
    return [self.map_actions_idx[t] for t in tokens]
    
  def convert_pms_to_indexs(self, tokens):
    return [self.map_pms_idx[t] for t in tokens]
    
  def convert_error_to_index(self, error):
  
    return self.error_type.index(error)
    
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
  
  def ___get_loader(self):
  
    def batchify_fn(list_data):
    
      input_words = []; input_valid_lens = []; input_segments = []
      
      target_words = []; target_valid_lens = []; target_segments = []
      
      target_actions = []; target_pms = []; list_input_texts = []
      
      error_embs = []; start_embs = []; end_embs = []; list_ids = []
      
      _list_target_texts = []
      
      if self.mode == 'train':
      
        for train_data in list_data:
        
          # print(train_data)
          
          list_ids.append(train_data['id'])
        
          str_input_text = train_data['input']
          if 'target' not in train_data:
            str_target_text = str_input_text
          else:
            str_target_text = train_data['target']
          
          # str_target_text = train_data['target']
          
          # print('input text => ', str_input_text)
          
          # print('target text => ', str_target_text)
          
          input_data = self.transformer([str_input_text])
          
          if len(str_input_text) > self.max_seq_len: # 超過長度
            continue
          
          target_data = self._transform_target(str_target_text)
           
          input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])
          target_word, target_valid_len = nd.array([target_data[0]]), nd.array([target_data[1]])

          # input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])
          target_segment = input_segment
          
          _list_target_texts.append(str_target_text)
          input_words.append(input_word.astype(np.float32)); input_valid_lens.append(input_valid_len.astype(np.float32)); input_segments.append(input_segment.astype(np.float32))
          target_words.append(target_word.astype(np.float32)); target_valid_lens.append(target_valid_len.astype(np.float32));
          target_segments.append(target_segment.astype(np.float32))
          # target_actions.append(target_action.astype(np.float32)); target_pms.append(target_pm.astype(np.float32));
          
          list_input_texts.append(str_input_text)
          
          _list_target_texts.append(str_target_text)
          
          # error_embs.append(error_emb)#; start_embs.append(start_emb); end_embs.append(end_emb)
          
        return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0), list_input_texts, _list_target_texts

      
      elif self.mode == 'test':
        
        for test_data in list_data:
          
          list_ids.append(test_data['id'])
          str_input_text = test_data['text']
          str_target_text = test_data['target']
          
          # print('input => ', str_input_text)
          
          # print('target => ', str_target_text)
          
          input_data = self.transformer([str_input_text])
          input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])
          input_words.append(input_word.astype(np.float32)); input_valid_lens.append(input_valid_len.astype(np.float32)); input_segments.append(input_segment.astype(np.float32))
          list_input_texts.append(str_input_text)
          _list_target_texts.append(str_target_text)
          
        return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), list_input_texts, _list_target_texts, list_ids
      # return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0)#, nd.concat(*target_actions, dim = 0), nd.concat(*target_pms, dim = 0), list_input_texts, list_target_texts
    self.dataset = SimpleDataset(self.data)
    if self.mode == 'test':
      shuffle = False
      last_batch = 'keep'
    else:
      shuffle = True
      last_batch = 'rollover'
      
    self.loader = DataLoader(self.dataset, batch_size = self.batch_size, batchify_fn = batchify_fn, shuffle = shuffle, last_batch = last_batch)
    
    return self.loader
  
    self.dataset = SimpleDataset(self.data)
    if self.mode == 'test':
      shuffle = False
      last_batch = 'keep'
    else:
      shuffle = True
      last_batch = 'rollover'
      
    self.loader = DataLoader(self.dataset, batch_size = self.batch_size, batchify_fn = batchify_fn, shuffle = shuffle, last_batch = last_batch)
    
    return self.loader
  
  def gen_eval_report(self, list_error):
  
    with open('prediction_csc15.txt', 'w+') as f:
      
      for error in list_error:
    
        f.write(error + '\n')
        
    
  def convert_to_report_format(self, list_id, list_error):#, list_start, list_end):
  
    list_result = []
  
    for (_id, _error) in zip(list_id, list_error):
    
      # print("_id ", _id)
      
      print('_error ', _error)
      
      # _error = _error.asnumpy()
    
      _erros = []
      
      total = len(_error); current = 0
      
      has_error = False
      
      while current < total:
      
        _error_type = _error[current]
        
        if _error_type > 0 and _error_type < 5:
          has_error = True
          start = current
          end = current
          while True:
            end += 1
            _error_type_next = _error[end]
            if not (_error_type_next > 0 and _error_type_next == _error_type):
              break
          current = end
          # if self.useDecoder:
            # _erros.append('{}, {}, {}'.format(start, end - 1, self.error_type[_error_type]))
          # else:
          
          # print('error type : ', _error_type)
          _erros.append('{}, {}, {}'.format(start + 1, end, self.error_type[_error_type]))
        else:
          current += 1
        
      if (len(_erros) == 0):
        _result = '{}, correct'.format(_id)
        list_result.append(_result)
      else:
        for _error in _erros:
          _result = '{}, {}'.format(_id, _error)
          list_result.append(_result)
    return list_result
    
  def offset(self, prediction_text, input_text):
  
    start_offset = None
  
    for it in input_text:
      if it in prediction_text:
        start_offset = prediction_text.index(it)
        
    return prediction_text[start_offset : start_offset + len(input_text)]
        
      
  def compare_input_prediction(self, id, prediction_text, input_text):
  
    print('prediction => ', prediction_text)
    print('input => ', input_text)
    print('len prediction => ', len(prediction_text))
    print('len input => ', len(input_text))
    
    if len(prediction_text) == len(input_text):
      self.compare_counter_same += 1
    else:
      self.compare_counter_diff +=1
      
    self.compare_counter +=1
    
    if (len(prediction_text) != len(input_text)):
    
      # print(prediction_text)
      
      # print(input_text)
    
      raise
    
      return '{}, {}'.format(id, 0)
      
    else :
    
      error_list = []
    
      for idx, (p, i) in enumerate(zip(prediction_text, input_text)):
      
        if (p != i):
        
          error_list.append([idx + 1, p])
          
      if len(error_list) == 0:
        print('same : ', float(self.compare_counter_same) / self.compare_counter)
        return '{}, {}'.format(id, 0)
      else:
        print('same : ', float(self.compare_counter_same) / self.compare_counter)
        return '{}, '.format(id) + ', '.join(['{}, {}'.format(e[0], e[1]) for e in error_list])
    
    score = sentence_bleu([[t for t in prediction_text]], [t for t in input_text])
    
    return 
    
    # score = sentence_bleu([t for t in prediction_text], [t for t in input_text])
    
    print(score)
    
    
  def get_loader(self):
  
    def batchify_fn(list_data):
    
      input_words = []; input_valid_lens = []; input_segments = []
      
      target_words = []; target_valid_lens = []; target_segments = []
      
      target_actions = []; target_pms = []; list_input_texts = []
      
      target_idxs = []
      
      error_embs = []; start_embs = []; end_embs = []; list_ids = []
      
      _list_target_texts = []
      
      if not self.config['csc_fixed']:
      
        if self.mode == 'train':
        
          for train_data in list_data:
          
            # print(train_data)
            list_ids.append(train_data['id'])
          
            str_input_text = train_data['input']
            # no target means no error 
            if 'target' not in train_data:
              str_target_text = str_input_text
            else:
              str_target_text = train_data['target']
            
            input_data = self.transformer([str_input_text])
            
            if len(str_input_text) > self.max_seq_len: # 超過長度
              continue
            target_data = self._transform_target(str_target_text)
            input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])
            target_word, target_valid_len = nd.array([target_data[0]]), nd.array([target_data[1]])
            # input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])
            target_segment = input_segment
            
            _list_target_texts.append(str_target_text)
            input_words.append(input_word.astype(np.float32)); input_valid_lens.append(input_valid_len.astype(np.float32)); input_segments.append(input_segment.astype(np.float32))
            target_words.append(target_word.astype(np.float32)); target_valid_lens.append(target_valid_len.astype(np.float32));
            target_segments.append(target_segment.astype(np.float32))
            # target_actions.append(target_action.astype(np.float32)); target_pms.append(target_pm.astype(np.float32));
            list_input_texts.append(str_input_text)
            
            _list_target_texts.append(str_target_text)
            # error_embs.append(error_emb)#; start_embs.append(start_emb); end_embs.append(end_emb)
            
          return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0), list_input_texts, _list_target_texts

        
        elif self.mode == 'test':
          
          for test_data in list_data:
            
            list_ids.append(test_data['id'])
            str_input_text = test_data['text']
            str_target_text = test_data['target']
            
            # print('input => ', str_input_text)
            
            # print('target => ', str_target_text)
            
            input_data = self.transformer([str_input_text])
            input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])
            input_words.append(input_word.astype(np.float32)); input_valid_lens.append(input_valid_len.astype(np.float32)); input_segments.append(input_segment.astype(np.float32))
            list_input_texts.append(str_input_text)
            _list_target_texts.append(str_target_text)
            
          return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), list_input_texts, _list_target_texts, list_ids
       
      else: # fixed length mode
    
        if self.mode == 'train':
          for train_data in list_data:
            list_ids.append(train_data['id'])
            str_input_text = train_data['input']
            # no target means no error 
            if 'target' not in train_data:
              str_target_text = str_input_text
            else:
              str_target_text = train_data['target']
            # print(str_target_text)
            # print(train_data['correction'])
            
            input_data = self.transformer([str_input_text])
            input_word, input_valid_len, input_segment = nd.array([input_data[0]]), nd.array([input_data[1]]), nd.array([input_data[2]])

            if input_word.shape[1] > self.max_seq_len: # 超過長度
              continue
            
            if 'correction' in train_data:
              target_idx = self.gen_fixed_length_correction_emb(input_word, train_data['correction'])
            else:
              target_idx = np.zeros([1, input_word.shape[1]])
              
            # target_idx = self.gen_fixed_length_correction_emb(input_word, train_data['correction'])
             
            # print(target_idx.shape)
            
            # raise
            
            # raise
              
            _list_target_texts.append(str_target_text)
            input_words.append(input_word.astype(np.float32)); input_valid_lens.append(input_valid_len.astype(np.float32)); input_segments.append(input_segment.astype(np.float32))
            # target_actions.append(target_action.astype(np.float32)); target_pms.append(target_pm.astype(np.float32));
            list_input_texts.append(str_input_text)
            
            target_idxs.append(target_idx)
            
            _list_target_texts.append(str_target_text)
            
          return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_idxs, dim = 0), list_input_texts, _list_target_texts
        
      # return nd.concat(*input_words, dim = 0), nd.concat(*input_valid_lens, dim = 0), nd.concat(*input_segments, dim = 0), nd.concat(*target_words, dim = 0), nd.concat(*target_valid_lens, dim = 0), nd.concat(*target_segments, dim = 0)#, nd.concat(*target_actions, dim = 0), nd.concat(*target_pms, dim = 0), list_input_texts, list_target_texts
    self.dataset = SimpleDataset(self.data)
    if self.mode == 'test':
      shuffle = False
      last_batch = 'keep'
    else:
      shuffle = True
      last_batch = 'rollover'
      
    self.loader = DataLoader(self.dataset, batch_size = self.batch_size, batchify_fn = batchify_fn, shuffle = shuffle, last_batch = last_batch)
    
    return self.loader

  
    
    
