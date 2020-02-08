from segmentator import Segmentator
from reviewdata import ReviewData
import mxnet as mx
from mxnet import gluon

class Segmentation(object):

  def __init__(self):
    
    self.actions = [None, 'add', 'remove', 'modify', '[START]']
    self.pms = [None, ':', '.', ',', '＜', '＞', '。', '?', '；', '、', "《", "》", '！', '，', '？', '「', '」', '[START]']
    self.segmentator = Segmentator(self.actions, self.pms)
    self.batch_size = 32
    self.data = ReviewData(self.segmentator.tokenizer, self.segmentator.transformer, self.batch_size, self.pms, self.actions)
    self.lr = 0.0001
    self.device = mx.gpu(0)
    
  def init_network(self):
  
    self.segmentator.initialize(mx.init.Xavier(), ctx = self.device)
    
    self.segmentator.collect_params().reset_ctx(self.device)
    
  def init_trainer(self):
  
    lr_decay_step = 1000
    lr_decay_rate = 0.9
    lr_scheduler = mx.lr_scheduler.FactorScheduler(lr_decay_step, lr_decay_rate)
    lr = self.lr
    
    optimizer = 'rmsprop'
    options = {
      'learning_rate': lr,
      'lr_scheduler' : lr_scheduler,
      'clip_gradient': 1,
       # 'momentum' : 0.9,
      'wd' : 0.0001
    }
    
    self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), optimizer, options)

    
  def train(self):
  
    self.init_network()
    
    self.init_trainer()
  
    self.loader = self.data.get_loader()
    
    for i, batch in enumerate(self.loader):
    
      for j in range(20000):
    
        nd_input_emb, nd_valid_len, nd_segment, nd_target_action, nd_target_pm, list_input_texts, list_target_texts = batch
        
        nd_input_emb = nd_input_emb.as_in_context(self.device)
        nd_valid_len = nd_valid_len.as_in_context(self.device)
        nd_segment = nd_segment.as_in_context(self.device)
        nd_target_action = nd_target_action.as_in_context(self.device)
        nd_target_pm = nd_target_pm.as_in_context(self.device)
        
        self.segmentator.train(nd_input_emb, nd_target_action, nd_target_pm, nd_segment, nd_valid_len, list_input_texts, list_target_texts, self.trainer)
  
    
  
    
    
    
    
    
    
    
if __name__ == '__main__':

  s = Segmentation()
  
  s.train()