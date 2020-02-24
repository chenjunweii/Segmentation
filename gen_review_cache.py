import os
import textract
import numpy as np
from utils import save_pickle
from reviewdata import ReviewData
from config import config
from segmentator import Segmentator

str_path = 'dataset'

segmentator = Segmentator(dict(), config)
data = ReviewData(segmentator.tokenizer, segmentator.transformer, segmentator.vocab_tgt, config, 'test')
    


def extract_from_paragraph(str_paragraph):

  return [line.strip() for line in str_paragraph.replace('\n', '').strip().split('<br />')]
  

def read_until(str_toStop, num_currentLine, list_allLines):

  str_paragraph = ''
  
  num_currentLine += 1
  
  line = list_allLines[num_currentLine]
  
  # print(line)
  
  while not str_toStop in line:
  
    str_paragraph = str_paragraph + line if len(line) > 0 else str_paragraph
  
    num_currentLine += 1
    
    line = list_allLines[num_currentLine]#.replace('<br />', '').strip()
  
    # print('line => ', line)
    
    # print(any([_toStop in line for _toStop in list_toStop]))
    
    # print(list_allLines[39])
    
    # print(num_currentLine)
    
  # print(str_paragraph)
    
  return str_paragraph, num_currentLine

def review_parser(str_filename, bool_splitString = False):
    
  f = open(str_filename)
  
  reviews = list()
  
  bool_isRecording = False
  
  list_lines = f.readlines()
  
  num_currLine = 0
    
  while num_currLine < len(list_lines):
  
    str_line = list_lines[num_currLine]
    
    str_paragraph = None
    
    if '一、圖書作者與內容簡介：' in str_line:
    
      str_paragraph, num_currLine = read_until('<BR>', num_currLine, list_lines)
      
      
    elif '二、內容摘錄：' in str_line:
    
      str_paragraph, num_currLine = read_until('<BR>', num_currLine, list_lines)
      
    elif '三、我的觀點：' in str_line:
    
      str_paragraph, num_currLine = read_until('<BR>', num_currLine, list_lines)
      
    elif '四、討論議題：' in str_line:
      
      str_paragraph, num_currLine = read_until('<BR>', num_currLine, list_lines)
      
    else:
    
      str_paragraph = None
      
    if str_paragraph is not None and len(str_paragraph) > 0:
     
      list_extractLines = extract_from_paragraph(str_paragraph)
      
      for str_extractLine in list_extractLines:
      
        if len(str_extractLine) > 10 and len(str_extractLine) <= config['int_max_length']:
      
          reviews.append(str_extractLine)
      
    num_currLine += 1
    
  return reviews
    
  # if bool_splitString:
  
if __name__ == '__main__':

  bool_splitString = True
  
  dataset = dict()
  
  for str_set, arr_datasets in config['dataset'].items():
  
    reviews = []
    
    for str_dataset in arr_datasets:

      str_path_dataset = os.path.join(str_path, str_dataset)
      
      print(str_path_dataset)
  
      for str_filename in os.listdir(str_path_dataset):
      
        str_filename = os.path.join(str_path_dataset, str_filename)
        
        print(str_filename)
        
        review = review_parser(str_filename, bool_splitString)
        
        review_errorize = []
        if str_set == 'test':
          for _review in review:
          
            str_input_text = data.pinyin_sampler.errorize_sentence(_review)
        
            str_input_text, _, _ = data._remove_pm(str_input_text)
            review_errorize.append([str_input_text, _review])
          
          review = review_errorize
        reviews.extend(review)
        
      
      
      
      
    dataset[str_set] = reviews
      
  save_pickle(dataset, 'reviews.cache')
      
  
    
