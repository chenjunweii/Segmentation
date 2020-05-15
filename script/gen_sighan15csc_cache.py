import os
import sys
sys.path.append(os.path.abspath('../'))
from html.parser import HTMLParser
from utils import save_pickle
import re
from random import shuffle

class Sighan15CSCTrainParser(HTMLParser):
  def __init__(self):
    super(Sighan15CSCTrainParser, self).__init__()
    self.data = []
  def handle_starttag(self, tag, attrs):
    self.tag = tag
    if tag == 'essay':
      self._data = dict()
      # self._data['input'] = dict()
      # self._data['target'] = dict()
      
    elif tag == 'mistake':
      self.mistake_attr = dict()
      for attr in attrs:
        self.mistake_attr[attr[0]] = attr[1]
        
    elif tag == 'passage':
      self.passage_attr = dict()
      for attr in attrs:
        self.passage_attr[attr[0]] = attr[1]
      
        
  def handle_endtag(self, tag):
  
    self.tag = None
  
    if tag == 'essay':
    
      for k, v in self._data.items():
        v['id'] = k
        self.data.append(v)
    elif tag == 'passage': 
      self._data[self.passage_attr['id']] = dict()
      self._data[self.passage_attr['id']]['input'] = self.passage_data
      
    elif tag == 'mistake':
    
      mistake_id = self.mistake_attr['id']
      
      input_text = self._data[mistake_id]['input']
      
      target_text = input_text.replace(self.wrong_data, self.correction_data)
      
      self._data[mistake_id]['target'] = target_text
      
  def handle_data(self, data):
  
    if self.tag == 'passage':
      self.passage_data = data
    elif self.tag == 'mistake':
      pass
    elif self.tag == 'wrong':
      self.wrong_data = data.strip()
    elif self.tag == 'correction':
      self.correction_data = data.strip()
  
      
      
class Sighan15CSCTestParser(object):

  def __init__(self, path):
  
    self.path = path
    
    # self.data = []
    
    # self.data["input"] = []
    # self.data['target'] = dict()
    
    self._data = dict()
    
  def get_data(self):
  
    return [v for v in self._data.values()]
    
  def parse(self, test_set):
  
    test_input, test_target = test_set
    
    with open(os.path.join(self.path, test_input)) as f_input, open(os.path.join(self.path, test_target)) as f_target:
    
      data_input = f_input.readlines()
      
      data_target = f_target.readlines()
      
      for _input in data_input:
      
        _input_split = _input.split('\t')
        
        _input_text = ''.join(_input_split[1:])
        
        _input_id = (_input_split[0].split('=')[1][:-1])
        
        self._data[_input_id] = {}
        
        self._data[_input_id]["id"] = _input_id.strip('\n')
        
        self._data[_input_id]["text"] = _input_text.strip('\n')
        
        self._data[_input_id]["target"] = []
        
        # self.data['input'].append({"id" : _input_id, 'text' : _input_text})
        
        # self.data['target'][_input_id] = []
        
      for _target in data_target:
      
        _target_id = _target.split(',')[0]
        
        assert(_target_id in self._data)
        
        self._data[_target_id]['target'].append(_target.strip('\n'))
        
parser = Sighan15CSCTrainParser()


train_set = ['Training/SIGHAN15_CSC_A2_Training.sgml', 'Training/SIGHAN15_CSC_B2_Training.sgml']

test_set = [('Test/SIGHAN15_CSC_TestInput.txt', 'Test/SIGHAN15_CSC_TestTruth.txt')]

parser_test = Sighan15CSCTestParser('../dataset/NCU_NLPLab_CSC/sighan8csc_release1.0')


for _train_set in train_set:
  with open(os.path.join('../dataset/NCU_NLPLab_CSC/sighan8csc_release1.0', _train_set), 'r') as f:
    print(_train_set)
    doc = ''.join((f.readlines()))
  out = parser.feed(doc)

for _test_set in test_set:
  parser_test.parse(_test_set)
  
  
data = dict()

data['train'] = parser.data
data['test'] = parser_test.get_data()


save_pickle(data, '../cache/sighancsc15.cache')
