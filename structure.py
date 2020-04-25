import re
import jieba
import jieba.posseg as pseg
from random import shuffle
from zhon import hanzi, cedict
from copy import deepcopy
import string

class Structure(object):

  def __init__(self):
  
    self.regex_sent = re.compile("""[{}{}0-9a-zA-Z].*[{}\n]""".format(cedict.all, hanzi.non_stops, hanzi.stops))
    
    self.regex_subsent = re.compile("""[{}{}0-9a-zA-Z\-]+[{}]""".format(cedict.all, hanzi.punctuation, hanzi.punctuation))
    
    self.regex_subsent_split = re.compile("""({})""".format('|'.join(hanzi.punctuation)))
          
  def randomize_word_order(self, text, min_len = 10):
    
    # list_sentences = re.findall(self.regex_sent, text)
    
    text_randomized = ''
        
    text_segs = re.split(self.regex_subsent_split, text)
    
    # raise
    
    for seg in text_segs:
    
      if len(seg) < min_len:
      
        text_randomized = text_randomized + seg
      
        continue
    
      words = list(jieba.cut(seg))
                  
      words_shuffled = deepcopy(words); shuffle(words_shuffled)
      
      text_randomized = text_randomized + ''.join(words_shuffled)
      
    assert(len(text) == len(text_randomized))
    
    return text_randomized
        
  def randomize_sentence_order(self, text):
  
    pass
      
    
    
    
  
    