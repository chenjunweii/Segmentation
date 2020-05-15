from html.parser import HTMLParser
from utils import save_pickle
import os
import re
from random import shuffle
class CGEDTrainParser(HTMLParser):
  def __init__(self):
    super(CGEDTrainParser, self).__init__()
    self.data = []
    self.split_data = dict()
    self.split_data['HSK'] = []
    self.split_data['TOCFL'] = []
  def handle_starttag(self, tag, attrs):
    self.tag = tag
    if tag == 'doc':
      self._data = dict()
      self._data["error"] = []
    elif tag == 'error':
      _error = dict()
      for k, v in attrs:
        _error[k] = v
      self._data["error"].append(_error)
    elif tag == 'text':
      for k, v in attrs:
        self._data[k] = v
  def handle_endtag(self, tag):
    if tag == 'doc':
      self.data.append(self._data)
      self.split_data[self.set].append(self._data)
      self._data = None
    self.tag = None
  def handle_data(self, data):
    if self.tag == 'text':
      # if len(data) > 256:
      
      #   print(data)
      self._data["input"] = data.strip('\n')
    elif self.tag == 'correction':
      # if len(data) > 256:
      
        # print(data)
      self._data["target"] = data.strip('\n')
      
      
class CGEDTestParser(object):

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
        
        self._data[_input_id]["id"] = _input_id
        
        self._data[_input_id]["text"] = _input_text
        
        self._data[_input_id]["target"] = []
        
        # self.data['input'].append({"id" : _input_id, 'text' : _input_text})
        
        # self.data['target'][_input_id] = []
        
      for _target in data_target:
      
        _target_id = _target.split(',')[0]
        
        assert(_target_id in self._data)
        
        self._data[_target_id]['target'].append(_target)
        
parser = CGEDTrainParser()

train_set = ['Training/CGED16_HSK_TrainingSet.txt', 'Training/CGED16_TOCFL_TrainingSet.txt']

test_set = [('Test/CGED16_HSK_Test_Input.txt', 'Test/CGED16_HSK_Test_Truth.txt'), ('Test/CGED16_TOCFL_Test_Input.txt', 'Test/CGED16_TOCFL_Test_Truth.txt')]

parser_test = CGEDTestParser('./cged/nlptea16cged_release1.0/')

for _test_set in test_set:

  parser_test.parse(_test_set)
  
CGED16 = dict()
CGED16['train'] = parser.data
CGED16['test'] = parser_test.get_data()

# print(parser_test.data)

for _train_set in train_set:

  with open(os.path.join('./cged/nlptea16cged_release1.0', _train_set), 'r') as f:
    doc = ''.join((f.readlines()))
    
  if 'HSK' in _train_set:
    parser.set = 'HSK'
    
  elif 'TOCFL' in _train_set:
    parser.set = 'TOCFL'
  
  out = parser.feed(doc)

print('train set : ', len(parser.data))
print('test set : ', len(CGED16['test']))

print(len(parser.split_data))
print(len(parser.split_data['HSK']))
print(len(parser.split_data['TOCFL']))

paper_dataset = parser.split_data['HSK'] + parser.split_data['TOCFL']
shuffle(paper_dataset)

paper_set_num = len(paper_dataset)
paper_train_set_num = int(paper_set_num * 0.8)

paper_train_set = paper_dataset[ : paper_train_set_num]
paper_test_set = paper_dataset[paper_train_set_num : ]

CGED16['paper-train'] = paper_train_set
CGED16['paper-test'] = paper_test_set

save_pickle(CGED16, 'CGED16.cache')

