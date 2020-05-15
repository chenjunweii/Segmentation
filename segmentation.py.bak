from segmentator import Segmentator
from reviewdata import ReviewData
from cgeddata import CGEDData
import mxnet as mx
import os
from mxnet import gluon
import string
from zhon import hanzi, zhuyin
from argparse import ArgumentParser
from config import config
from utils import load_pickle
from interactive import shell
from tqdm import tqdm
import cmd
from mxnet_utils.model import load_latest_checkpoint, save_gluon_model, load_pretrained_model, load_pretrained_model_only_same_shape, load_pretrained_model_only_same_shape_sup  
class Segmentation(object):

  def __init__(self, args):
  
    self.args = args    
    self.lr = 0.0001
    self.device = ([mx.gpu(0), mx.gpu(1)] if not args.cpu else [mx.cpu(0)]) if args.train else [mx.gpu(0)]
    self.lr_decay_step = 1000
    self.lr_decay_epoch = 2
    self.lr_decay_rate_epoch = 0.9
    self.lr_decay_rate = 1
    self.save_freq = 1 # epoch
    self.save_freq_step = 5000
    self.arch_name = config['arch_name']
    self.arch_path = os.path.join('model', self.arch_name)
    self.mode = 'train' if self.args.train else 'test'
    self.batch_size = config[self.mode]['batch_size']

    self.config = config
    
    self.segmentator = Segmentator(self.args, self.config)
    
    if args.dataset == 'CGED16':
    
      self.data = CGEDData(self.segmentator.tokenizer, self.segmentator.transformer, self.config, self.mode, useDecoder = self.args.decoder, args = args)
      
      self.loader = self.data.get_loader()
      
    elif args.dataset == 'paper':
    
      self.cgeddata = CGEDData(self.segmentator.tokenizer, self.segmentator.transformer, self.config, self.mode, useDecoder = self.args.decoder, args = args)
      
      self.reviewdata = ReviewData(self.segmentator.tokenizer, self.segmentator.transformer, self.segmentator.vocab_tgt, self.config, self.mode)
    
      # enumerator_loader = enumerate(self.loader)
      
      # enumerator_loader.__next__()
      
      # raise
    
    else:
      self.data = ReviewData(self.segmentator.tokenizer, self.segmentator.transformer, self.segmentator.vocab_tgt, self.config, self.mode)
      self.data_val = ReviewData(self.segmentator.tokenizer, self.segmentator.transformer, self.segmentator.vocab_tgt, self.config, 'test')
    
  def init_network(self):
    self.segmentator.initialize(mx.init.Xavier(), ctx = self.device)
    
    if self.args.dataset != 'CGED16':
      emb_pretrained = self.segmentator._collect_params_with_prefix()['encoder.word_embed.0.weight'].data()[:len(self.segmentator.vocab_tgt)]
      self.segmentator.collect_params().reset_ctx(self.device)
      self.segmentator._collect_params_with_prefix()['emb_tgt.0.weight']._load_init(emb_pretrained, self.device)
    else:
      self.segmentator.collect_params().reset_ctx(self.device)
      
    if self.args.pretrain:
      return load_pretrained_model_only_same_shape(self.segmentator, 'model/bert_128_1/0006-0.params', self.device)
    else:
      return load_latest_checkpoint(self.segmentator, self.arch_path, self.device)
    # if os.path.isdir('model/bert'):
    
  def init_trainer(self, epoch_start):
    self.lr_scheduler = mx.lr_scheduler.FactorScheduler(self.lr_decay_step, self.lr_decay_rate)
    self.optimizer = 'lamb'
    self.options = {
      'learning_rate': self.lr * (self.lr_decay_rate_epoch ** int(epoch_start / float(self.lr_decay_epoch))),
      'lr_scheduler' : self.lr_scheduler,
      'clip_gradient': 0.1,
       # 'momentum' : 0.9,
      'wd' : 0.0001
    }
    
    self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), self.optimizer, self.options)

  def train(self):
  
    epoch_start, _ = self.init_network()
    epoch_start += 1
    
    self.init_trainer(epoch_start)
    self.loader = self.data.get_loader()
    self.loader_val = self.data_val.get_loader()
    
    self.num_samples = len(self.data.data)
    self.segmentator.hybridize()
    
    list_input_texts_test = None
    list_target_texts_test = None
    
    enumerator_val = enumerate(self.loader_val)
    
    # raise
    
    # for _, _batch_test_0 in enumerate(self.loader_val):
    
    #   list_input_texts_test = _batch_test_0[6]
    #   list_target_texts_test = _batch_test_0[7]
      
    #   break
    # raise
    
    for e in range(epoch_start, epoch_start + 100):
      progress = tqdm(total = self.num_samples)
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
        
        # print('input text => ', list_input_texts[0])
        
        # print("target text => ", list_target_texts[0])
        
        # raise
    
        loss = self.segmentator.train(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                nd_target_word_idx, nd_target_valid_len, nd_target_segment,
                                list_input_texts[:_batch_size], list_target_texts[:_batch_size], self.device, _batch_size, self.trainer)
        if (i % 100 == 0):
        
          print("=" * 10)
          
          print('Epoch {} Step {} => {}, LR : {}'.format(e, i, loss, self.trainer.learning_rate))
          
          _, batch_test = enumerator_val.__next__()
          list_input_texts_test = batch_test[6]
          list_target_texts_test = batch_test[7]
          
          print('Input => ', list_input_texts_test[0])
          print('Target => ', list_target_texts_test[0])
          
          text = self.segmentator.run(list_input_texts_test[0], self.device)
          
        progress.update(self.batch_size)
          
      if e % self.save_freq == 0:
        save_gluon_model(self.segmentator, self.arch_path, e, 0) # use main     dataset
      elif i % self.save_freq_step == 0:
        save_gluon_model(self.segmentator, self.arch_path, e - 1, i) # use main     dataset
        
      
        
      
      self.options['learning_rate'] = self.trainer.learning_rate * self.lr_decay_rate_epoch
      
      self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), self.optimizer, self.options)

  def test(self):
  
    self.segmentator.hybridize()
    self.loader = self.data.get_loader()
    for i, batch in enumerate(self.loader):      
      nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, list_input_texts, list_target_texts = batch
        
    
    pass
    
  def run(self, text):
  
    self.init_network()
    self.segmentator.hybridize()
    self.segmentator.run(text, self.device)
      
  def interact(self):
  
    self.init_network()
    self.segmentator.hybridize()
    shell(self.segmentator, self.arch_path, self.device).cmdloop()
      
    
    
  def train_CGED(self):
    epoch_start, _ = self.init_network()
    epoch_start += 1
    
    self.init_trainer(epoch_start)
    self.loader = self.data.get_loader()
    # self.loader_val = self.data_val.get_loader()
    
    self.num_samples = len(self.data.data)
    self.segmentator.hybridize()

    for e in range(epoch_start, epoch_start + 10):
      progress = tqdm(total = self.num_samples)
      for i, batch in enumerate(self.loader):      
        nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_error_idx, list_input_texts, list_target_texts = batch
        _batch_size = int(nd_input_word_idx.shape[0] / len(self.device)) * len(self.device)
        
        nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx[:_batch_size], self.device)
        nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len[:_batch_size], self.device)
        nd_input_segment = gluon.utils.split_and_load(nd_input_segment[:_batch_size], self.device)
        
        nd_error_idx = gluon.utils.split_and_load(nd_error_idx[:_batch_size], self.device)
        # nd_start_idx = gluon.utils.split_and_load(nd_start_idx[:_batch_size], self.device)
        # nd_end_idx = gluon.utils.split_and_load(nd_end_idx[:_batch_size], self.device)
         
        if self.args.decoder:
          
          loss = self.segmentator.train_CGEDDecoder(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                nd_error_idx, self.device, _batch_size, self.trainer)
                                
          # if (i % 100 == 0):
        
          #   print("=" * 10)
            
          #   print('Epoch {} Step {} => {}, LR : {}'.format(e, i, loss, self.trainer.learning_rate))
            
          #   _, batch_test = enumerator_val.__next__()
          #   list_input_texts_test = batch_test[6]
          #   list_target_texts_test = batch_test[7]
            
          #   print('Input => ', list_input_texts_test[0])
          #   print('Target => ', list_target_texts_test[0])
            
          #   text = self.segmentator.run(list_input_texts_test[0], self.device)
          
        else:
        
          loss = self.segmentator.train_CGED(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                nd_error_idx, self.device, _batch_size, self.trainer)
    

        progress.update(self.batch_size)
        
        if (i % 10 == 0):
        
          loss = sum([_loss.asnumpy().mean() for _loss in loss])# / _batch_size
          print('[*] Loss : {}, Lr : {}'.format(loss, self.trainer.learning_rate))
        
      if e % self.save_freq == 0:
        save_gluon_model(self.segmentator, self.arch_path, e, 0) # use main     dataset
      elif i % self.save_freq_step == 0:
        save_gluon_model(self.segmentator, self.arch_path, e - 1, i) # use main     dataset

      self.options['learning_rate'] = self.trainer.learning_rate * self.lr_decay_rate_epoch
      
      self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), self.optimizer, self.options)

  def test_CGED(self):
  
    self.init_network()
    self.loader = self.data.get_loader()
    self.num_samples = len(self.data.data)
    self.segmentator.hybridize()

    progress = tqdm(total = self.num_samples)
    
    
    list_result = []
    for i, batch in enumerate(self.loader):      
      nd_input_word_idx, nd_input_valid_len, nd_input_segment, list_ids, list_input_texts, list_target_texts = batch
      _batch_size = int(nd_input_word_idx.shape[0] / len(self.device)) * len(self.device)
      
      nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx[:_batch_size], self.device)
      nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len[:_batch_size], self.device)
      nd_input_segment = gluon.utils.split_and_load(nd_input_segment[:_batch_size], self.device)
      
      
      if args.decoder:
        _predict_error_idx = self.segmentator.test_CGEDDecoder(nd_input_word_idx, nd_input_valid_len, nd_input_segment, self.device)
      else:
        _predict_error_idx = self.segmentator.test_CGED(nd_input_word_idx, nd_input_valid_len, nd_input_segment, self.device)
        
      _list_result = self.data.convert_to_report_format(list_ids, _predict_error_idx)
  
      list_result.extend(_list_result)
      
      progress.update(self.batch_size)
      
    self.data.gen_eval_report(list_result)
    
  def train_paper(self):
  
    epoch_start, _ = self.init_network()
    epoch_start += 1
    
    self.init_trainer(epoch_start)
    self.loader = self.data.get_loader()
    # self.loader_val = self.data_val.get_loader()
    
    self.num_samples = len(self.data.data)
    self.segmentator.hybridize()
  
    pass
  
    

parser = ArgumentParser()
parser.add_argument("--train", dest = 'train', action = 'store_true')
parser.add_argument("--test", dest = 'test', action = 'store_true')
parser.add_argument("--run", dest = 'run', action = 'store_true')
parser.add_argument("--decoder", dest = 'decoder', action = 'store_true')
parser.add_argument("--interact", dest = 'interact', action = 'store_true')
parser.add_argument("--pretrain", dest = 'pretrain', action = 'store_true')
parser.add_argument("--dataset", dest = 'dataset')


parser.add_argument("--text", dest = 'text')
parser.add_argument("--use-pretrained", dest = 'use_pretrained', action = 'store_true')
parser.add_argument("--use-tc", dest = 'use_tc', action = 'store_true')
parser.add_argument("--cpu", dest = "cpu", action = 'store_true')
parser.add_argument("-ag", '--accumulate-gradient', dest = 'ag', action = 'store_true')
parser.add_argument("--paper", dest = 'paper', action = 'store_true')

args = parser.parse_args()
    
    
if __name__ == '__main__':

  s = Segmentation(args)
  
  if args.train:
  
    if args.dataset == 'CGED16':
      s.train_CGED()
    elif args.dataset == 'paper':
      s.train_paper()
    else:
      s.train()
    
  elif args.test:
    if args.dataset == 'CGED16':
      s.test_CGED()
    elif args.dataset == 'paper':
      s.test_paper()
    else:
      s.test()
    
  elif args.run:
  
    s.run(args.text)
    
  elif args.interact:
  
    s.interact()
