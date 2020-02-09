from segmentator import Segmentator
from reviewdata import ReviewData
import mxnet as mx
import os
from mxnet import gluon
from mxnet_utils.model import load_latest_checkpoint, save_gluon_model, load_pretrained_model, load_pretrained_model_only_same_shape, load_pretrained_model_only_same_shape_sup  
class Segmentation(object):

  def __init__(self):
    
    self.actions = [None, 'add', 'remove', 'modify', '[START]']
    self.pms = [None, ':', '.', ',', '＜', '＞', '。', '?', '；', '、', "《", "》", '！', '，', '？', '「', '」', '[START]']
    self.max_seq_len = 128
    self.batch_size = 32
    self.lr = 0.0005
    self.device = [mx.gpu(0), mx.gpu(1)]
    self.lr_decay_step = 1000
    self.lr_decay_epoch = 2
    self.lr_decay_rate_epoch = 0.9
    self.lr_decay_rate = 1
    self.save_freq = 5
    self.arch_name = 'bert_100'
    self.arch_path = os.path.join('model', self.arch_name)
    
    self.segmentator = Segmentator(self.actions, self.pms, self.max_seq_len)
    self.data = ReviewData(self.segmentator.tokenizer, self.segmentator.transformer, self.batch_size, self.pms, self.actions, self.max_seq_len)
    
  def init_network(self):
    self.segmentator.initialize(mx.init.Xavier(), ctx = self.device)
    self.segmentator.collect_params().reset_ctx(self.device)
    
    return load_pretrained_model_only_same_shape(self.segmentator, 'model/bert/0023-0.params', self.device)
    return load_latest_checkpoint(self.segmentator, self.arch_path, self.device)
    # if os.path.isdir('model/bert'):
      
    
  def init_trainer(self, epoch_start):
    self.lr_scheduler = mx.lr_scheduler.FactorScheduler(self.lr_decay_step, self.lr_decay_rate)
    self.optimizer = 'lamb'
    self.options = {
      'learning_rate': self.lr * (self.lr_decay_rate_epoch ** int(epoch_start / float(self.lr_decay_epoch))),
      'lr_scheduler' : self.lr_scheduler,
      'clip_gradient': 1,
       # 'momentum' : 0.9,
      #'wd' : 0.0001
    }
    
    self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), self.optimizer, self.options)

    
  def train(self):
  
    epoch_start, _ = self.init_network()
    epoch_start += 1
    
    self.init_trainer(epoch_start)
    self.loader = self.data.get_loader()
    
    for e in range(epoch_start, epoch_start + 100):
      for i, batch in enumerate(self.loader):      
        nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, list_input_texts, list_target_texts = batch
        
        # nd_input_word_idx = nd_input_word_idx.as_in_context(self.device)
        # nd_input_valid_len = nd_input_valid_len.as_in_context(self.device)
        # nd_input_segment = nd_input_segment.as_in_context(self.device)
        
        # nd_target_word_idx = nd_target_word_idx.as_in_context(self.device)
        # nd_target_valid_len = nd_target_valid_len.as_in_context(self.device)
        # nd_target_segment = nd_target_segment.as_in_context(self.device)
        
        nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx, self.device)
        nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len, self.device)
        nd_input_segment = gluon.utils.split_and_load(nd_input_segment, self.device)
        
        nd_target_word_idx = gluon.utils.split_and_load(nd_target_word_idx, self.device)
        nd_target_valid_len = gluon.utils.split_and_load(nd_target_valid_len, self.device)
        nd_target_segment = gluon.utils.split_and_load(nd_target_segment, self.device)
        
        
    
        loss, text = self.segmentator.train(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                nd_target_word_idx, nd_target_valid_len, nd_target_segment,
                                list_input_texts, list_target_texts, self.device, self.batch_size, self.trainer)
        if (i % 10 == 0):
          print('Epoch {} => {}, LR : {}'.format(e, loss, self.trainer.learning_rate))
          print('Text => {}'.format(text))
          
      if e % self.save_freq:
        save_gluon_model(self.segmentator, self.arch_path, e, 0) # use main     dataset
      
      self.options['learning_rate'] = self.trainer.learning_rate * self.lr_decay_rate_epoch
      
      self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), self.optimizer, self.options)


    
    
    
    
    
    
    
if __name__ == '__main__':

  s = Segmentation()
  
  s.train()