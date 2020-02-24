from segmentator import Segmentator
from reviewdata import ReviewData
import mxnet as mx
import os
from mxnet import gluon
import string
from zhon import hanzi, zhuyin
from argparse import ArgumentParser
from utils import load_pickle
from config import config
from utils import load_pickle
from mxnet_utils.model import load_latest_checkpoint, save_gluon_model, load_pretrained_model, load_pretrained_model_only_same_shape, load_pretrained_model_only_same_shape_sup  
class Segmentation(object):

  def __init__(self, args):
  
    self.args = args    
    self.lr = 0.0005
    self.device = [mx.gpu(0), mx.gpu(1)]
    self.lr_decay_step = 1000
    self.lr_decay_epoch = 2
    self.lr_decay_rate_epoch = 0.9
    self.lr_decay_rate = 1
    self.save_freq = 5
    self.arch_name = 'bert_256_errorize'
    self.arch_path = os.path.join('model', self.arch_name)
    self.mode = 'train' if self.args.train else 'test'
    self.batch_size = config[self.mode]['batch_size']

    self.config = config
    
    self.segmentator = Segmentator(self.args, self.config)
    self.data = ReviewData(self.segmentator.tokenizer, self.segmentator.transformer, self.segmentator.vocab_tgt, self.config, self.mode)
    
  def init_network(self):
    self.segmentator.initialize(mx.init.Xavier(), ctx = self.device)
    emb_pretrained = self.segmentator._collect_params_with_prefix()['encoder.word_embed.0.weight'].data()[:len(self.segmentator.vocab_tgt)]
    self.segmentator.collect_params().reset_ctx(self.device)
    self.segmentator._collect_params_with_prefix()['emb_tgt.0.weight']._load_init(emb_pretrained, self.device)
    return load_pretrained_model_only_same_shape(self.segmentator, 'model/bert_128_errorize/0003-0.params', self.device)
    # return load_latest_checkpoint(self.segmentator, self.arch_path, self.device)
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
    
    self.segmentator.hybridize()
    
    for e in range(epoch_start, epoch_start + 100):
      for i, batch in enumerate(self.loader):      
        nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, list_input_texts, list_target_texts = batch
        
        # nd_input_word_idx = nd_input_word_idx.as_in_context(self.device)
        # nd_input_valid_len = nd_input_valid_len.as_in_context(self.device)
        # nd_input_segment = nd_input_segment.as_in_context(self.device)
        
        # nd_target_word_idx = nd_target_word_idx.as_in_context(self.device)
        # nd_target_valid_len = nd_target_valid_len.as_in_context(self.device)
        # nd_target_segment = nd_target_segment.as_in_context(self.device)
        
        _batch_size = int(nd_input_word_idx.shape[0] / len(self.device)) * len(self.device)
        
        nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx[:_batch_size], self.device)
        nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len[:_batch_size], self.device)
        nd_input_segment = gluon.utils.split_and_load(nd_input_segment[:_batch_size], self.device)
        
        nd_target_word_idx = gluon.utils.split_and_load(nd_target_word_idx[:_batch_size], self.device)
        nd_target_valid_len = gluon.utils.split_and_load(nd_target_valid_len[:_batch_size], self.device)
        nd_target_segment = gluon.utils.split_and_load(nd_target_segment[:_batch_size], self.device)
        
        
    
        loss, text = self.segmentator.train(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                nd_target_word_idx, nd_target_valid_len, nd_target_segment,
                                list_input_texts[:_batch_size], list_target_texts[:_batch_size], self.device, _batch_size, self.trainer)
        if (i % 10 == 0):
          print('Epoch {} => {}, LR : {}'.format(e, loss, self.trainer.learning_rate))
          print('Text => {}'.format(text))
          
      if e % self.save_freq:
        save_gluon_model(self.segmentator, self.arch_path, e, 0) # use main     dataset
      
      self.options['learning_rate'] = self.trainer.learning_rate * self.lr_decay_rate_epoch
      
      self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), self.optimizer, self.options)

  def test(self):
  
    self.segmentator.hybridize()
    self.loader = self.data.get_loader()

    
    for i, batch in enumerate(self.loader):      
      nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, list_input_texts, list_target_texts = batch
        
    
    pass

parser = ArgumentParser()
parser.add_argument("--train", dest = 'train', action = 'store_true')
parser.add_argument("--test", dest = 'test', action = 'store_true')
parser.add_argument("--use-pretrained", dest = 'use_pretrained', action = 'store_true')
parser.add_argument("--use-tc", dest = 'use_tc', action = 'store_true')


parser.add_argument("-ag", '--accumulate-gradient', dest = 'ag', action = 'store_true')
args = parser.parse_args()
    
    
if __name__ == '__main__':

  s = Segmentation(args)
  
  if args.train:
  
    s.train()
    
  else:
  
    s.test()