import re
from zhon import hanzi
import numpy as np
from random import choice
from pypinyin import pinyin, lazy_pinyin, Style
import jieba
from termcolor import colored
from opencc import OpenCC
cc = OpenCC('t2s')  # convert from Simplified Chinese to Traditional Chinese


class PinYinSampler(object):

  def __init__(self, tgt_vocab_list, word_vocab_list, options):
         
    self.re = dict()

    self.re['alphabet'] = re.compile('[a-zA-Z]')

    self.char_vowel_table = dict()
    
    self.options = options
        
    self.word_vowel_table = dict()

    for c in tgt_vocab_list:

      if self.isolation(c):

        continue
         
      v = '/'.join(lazy_pinyin(c)[0])
      
      if v not in self.char_vowel_table:
  
        self.char_vowel_table[v] = []
  
      self.char_vowel_table[v].append(c)

    for vocab in word_vocab_list:

      if len(self.re['alphabet'].findall(vocab)) > 0 or len(vocab) == 1:
                
        continue

      v = '/'.join([_w for _w in lazy_pinyin(vocab)]) # word vowel

      if v not in self.word_vowel_table:

        self.word_vowel_table[v] = []
        
      if vocab not in self.word_vowel_table[v]:

        self.word_vowel_table[v].append(vocab)
      
  def isolation(self, c):
    if c in hanzi.non_stops:
      return True
    elif self.re['alphabet'].match(c) is not None:
      return True
    else:
      return False
      
  def sample_same_vowel_char(self, c):

    same_vowels = self.list_same_vowel_char(c)

    return choice(same_vowels)

  def sample_same_vowel_word(self, w):
        
    same_vowels = self.list_same_vowel_word(w)
    
    # print('vowels -> ', same_vowels)

    return choice(same_vowels)

  def list_same_vowel_word(self, w):

    v = '/'.join([_w for _w in lazy_pinyin(w)]) # word vowel
    
    # print('v => ', v)

    if v not in self.word_vowel_table:

      return [w]

    return self.word_vowel_table[v]

  def list_same_vowel_char(self, c):

    if self.isolation(c):

      return [c]

    v = '/'.join(lazy_pinyin(c)[0])

    if v not in self.char_vowel_table:

      return [c]

    return self.char_vowel_table[v]
        
  def errorize_sentence(self, target_text):
    
    error_text = ''
    
    for token in list(jieba.cut(target_text)):
    
      if len(token) == 1 and np.random.ranf() < self.options['float_char_error_rate']:
      
        error_text = error_text + self.sample_same_vowel_char(token)
        
      elif len(token) > 1 and np.random.ranf() < self.options['float_word_error_rate']:
      
      # if True:
      
        error_token = self.sample_same_vowel_word(token)
        
        if (error_token) == token: # 如果沒找到 same_vowel_word 就拆開做
        
          random_token = ''
        
          for t in token:
          
            if np.random.ranf() < self.options['float_char_error_rate']:
      
              random_token = random_token + self.sample_same_vowel_char(t)
              
            else:
            
              random_token = random_token + t
              
          error_text = error_text + random_token            
          
        else:
        
          error_text = error_text + error_token
        
        
        
        # print('{} => {}'.format(token, self.sample_same_vowel_word(token)))
        
      else:
      
        error_text = error_text + token
        
    return error_text
