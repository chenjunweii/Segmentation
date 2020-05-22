from segmentator import Segmentator
from data.reviewdata import ReviewData
from data.csc15data import SighanCSC15Data
from data.csc14data import SighanCSC14Data
import numpy as np
from data.cgeddata import CGEDData
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
from mxboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu



from mxnet_utils.model import load_latest_checkpoint, save_gluon_model, load_pretrained_model, load_pretrained_model_only_same_shape, load_pretrained_model_only_same_shape_sup  
class Segmentation(object):

  def __init__(self, args):
  
    self.args = args    
    self.lr = 0.0005
    self.device = [mx.gpu(0)]#([mx.gpu(0), mx.gpu(1)] if not args.cpu else [mx.cpu(0)]) if args.train else [mx.gpu(0)]
    self.test_device = [mx.gpu(0)]
    self.lr_decay_step = 5000
    # self.lr_decay_epoch = 2
    self.lr_decay_rate_epoch = 0.9
    self.lr_decay_rate = 0.9
    self.save_freq = 1 # epoch
    self.save_freq_step = 5000
    self.arch_name = config['arch_name']
    self.arch_path = os.path.join('model', self.arch_name)
    self.mode = 'train' if self.args.train else 'test'
    self.batch_size = config[self.mode]['batch_size']
    self.config = config
    self.segmentator = Segmentator(self.args, self.config)
    self.cgeddata = CGEDData(self.segmentator.tokenizer, self.segmentator.transformer, self.config, self.mode, useDecoder = self.args.decoder, args = args)
    
    self.csc15data = SighanCSC15Data(self.segmentator.tokenizer, self.segmentator.transformer, self.config, self.mode, args = args, vocab_tgt = self.segmentator.vocab_tgt)
    self.csc14data = SighanCSC14Data(self.segmentator.tokenizer, self.segmentator.transformer, self.config, self.mode, args = args, vocab_tgt = self.segmentator.vocab_tgt)
    
    if args.dataset != 'CGED16':
      self.reviewdata = ReviewData(self.segmentator.tokenizer, self.segmentator.transformer, self.segmentator.vocab_tgt, self.config, self.mode)
      self.reviewdata_val = ReviewData(self.segmentator.tokenizer, self.segmentator.transformer, self.segmentator.vocab_tgt, self.config, 'test')

  def init_network(self):
    self.segmentator.initialize(mx.init.Xavier(), ctx = self.device)
    
    if self.args.dataset != 'CGED16':
      emb_pretrained = self.segmentator._collect_params_with_prefix()['encoder.word_embed.0.weight'].data()[:len(self.segmentator.vocab_tgt)]
      self.segmentator.collect_params().reset_ctx(self.device)
      self.segmentator._collect_params_with_prefix()['emb_tgt.0.weight']._load_init(emb_pretrained, self.device)
    else:
      self.segmentator.collect_params().reset_ctx(self.device)
      
    if self.args.pretrain:
      return load_pretrained_model_only_same_shape(self.segmentator, 'model/bert_multi_full/0000140000-0.params', self.device)
    else:
      return load_latest_checkpoint(self.segmentator, self.arch_path, self.device)
    # if os.path.isdir('model/bert'):
    
  def init_trainer(self, step_start):
    self.lr_scheduler = mx.lr_scheduler.FactorScheduler(self.lr_decay_step, self.lr_decay_rate)
    self.optimizer = 'lamb'
    self.options = {
      'learning_rate': self.lr * (self.lr_decay_rate_epoch ** int(step_start / float(self.lr_decay_step))),
      'lr_scheduler' : self.lr_scheduler,
      'clip_gradient': 0.1,
       # 'momentum' : 0.9,
      'wd' : 0.0001
    }
    
    self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), self.optimizer, self.options)

  def train(self):
  
    step_start, _ = self.init_network()
    
    self.init_trainer(step_start)
    
    self.sw = SummaryWriter(logdir = 'logs/' + self.config['arch_name'], flush_secs=5)
    
    
    # self.cgedloader = self.cgeddata.get_loader()
    self.reviewloader = self.reviewdata.get_loader()
    # self.csc15loader = self.csc15data.get_loader()
    # self.csc14loader = self.csc14data.get_loader()
    self.reviewloader_val = self.reviewdata_val.get_loader()
 
    
    # self.int_cged_samples = len(self.cgeddata.data)
    self.int_review_samples = len(self.reviewdata.data)
    # self.int_csc15_samples = len(self.csc15data.data)
    # self.int_csc14_samples = len(self.csc14data.data)
    
    self.segmentator.hybridize()
    
    list_input_texts_test = None
    list_target_texts_test = None
    
    # self.cged_enumerator = enumerate(self.cgedloader)
    # self.cged_epoch = 0
    self.review_epoch = 0
    # self.csc15_epoch = 0
    # self.csc14_epoch = 0
    self.review_enumerator = enumerate(self.reviewloader)
    # self.csc15_enumerator = enumerate(self.csc15loader)
    # self.csc14_enumerator = enumerate(self.csc14loader)
    self.review_enumerator_val = enumerate(self.reviewloader_val)
    
    
    # progress_cged = tqdm(total = self.int_cged_samples, desc = 'cged')
    progress_review = tqdm(total = self.int_review_samples, desc = 'review')
    # progress_csc15 = tqdm(total = self.int_csc15_samples, desc = 'csc15')
    # progress_csc14 = tqdm(total = self.int_csc14_samples, desc = 'csc14')
    
    
    # for intialize parameter in network for multi-task
    # i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment,  nd_pm_error_idx, nd_pm_add_idx, nd_pm_remove_idx, list_input_texts, list_target_texts) = next(self.review_enumerator, (-1, [None] * 11))
    # nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx[:2], self.device)
    # nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len[:2], self.device)
    # nd_input_segment = gluon.utils.split_and_load(nd_input_segment[:2], self.device)
    
    # nd_target_word_idx = gluon.utils.split_and_load(nd_target_word_idx[:2], self.device)
    # nd_target_valid_len = gluon.utils.split_and_load(nd_target_valid_len[:2], self.device)
    # nd_target_segment = gluon.utils.split_and_load(nd_target_segment[:2], self.device)
    # self.segmentator.initialize_full_network(nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, self.device)
    
    for _step in range(step_start, step_start + 30001):
    
      # try: # 為了解決 batch 剩 1 的問題，但是要 debug 時很難 debug
      _dataset = np.random.choice(['review', 'cged', 'csc15', 'csc14'], p = [1, 0, 0, 0])
      
      if (_step % self.config['val_freq'] == 0 or _step % 100 == 0) and _step != 0:
        _dataset = 'review'
    
      if _dataset == 'cged':
        # i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, list_ids, list_input_texts, list_target_texts) = self.cged_enumerator.__next__()
        i, ( nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_error_idx, list_input_texts, list_target_texts) = next(self.cged_enumerator, (-1, [None] * 6))
        if i == -1:
          self.cgedloader = self.cgeddata.get_loader()
          self.cged_epoch += 1
          self.cged_enumerator = enumerate(self.cgedloader)
          i, ( nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_error_idx, list_input_texts, list_target_texts) = next(self.cged_enumerator, (-1, [None] * 6))
          progress_cged = tqdm(total = self.int_cged_samples, desc = 'cged')
        _batch_size = int(nd_input_word_idx.shape[0] / len(self.device)) * len(self.device)
        
        nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx[:_batch_size], self.device)
        nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len[:_batch_size], self.device)
        nd_input_segment = gluon.utils.split_and_load(nd_input_segment[:_batch_size], self.device)
        
        nd_error_idx = gluon.utils.split_and_load(nd_error_idx[:_batch_size], self.device)
        
        loss = self.segmentator.train_CGED(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                nd_error_idx, self.device, _batch_size, self.trainer)
        progress_cged.update(_batch_size)
        
        if (_step % 100 == 0):
          print('CGED Loss => {}, Epoch => {}, Lr => {}'.format(loss, self.cged_epoch, self.trainer.learning_rate))
        
      elif _dataset == 'review':
        i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, nd_pm_error_idx, nd_pm_add_idx, nd_pm_remove_idx, list_input_texts, list_target_texts) = next(self.review_enumerator, (-1, [None] * 11))
        if i == -1:
          self.reviewloader = self.reviewdata.get_loader()
          self.review_epoch += 1
          self.review_enumerator = enumerate(self.reviewloader)
          i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, nd_pm_error_idx, nd_pm_add_idx, nd_pm_remove_idx, list_input_texts, list_target_texts) = next(self.review_enumerator, (-1, [None] * 11))
          progress_review = tqdm(total = self.int_review_samples, desc = 'review')
          
        _batch_size = int(nd_input_word_idx.shape[0] / len(self.device)) * len(self.device)
        nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx[:_batch_size], self.device)
        nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len[:_batch_size], self.device)
        nd_input_segment = gluon.utils.split_and_load(nd_input_segment[:_batch_size], self.device)
        
        
        
        nd_pm_error_idx = gluon.utils.split_and_load(nd_pm_error_idx[:_batch_size], self.device)
        nd_pm_add_idx = gluon.utils.split_and_load(nd_pm_add_idx[:_batch_size], self.device)
        nd_pm_remove_idx = gluon.utils.split_and_load(nd_pm_remove_idx[:_batch_size], self.device)
        nd_target_word_idx = gluon.utils.split_and_load(nd_target_word_idx[:_batch_size], self.device)
        nd_target_valid_len = gluon.utils.split_and_load(nd_target_valid_len[:_batch_size], self.device)
        nd_target_segment = gluon.utils.split_and_load(nd_target_segment[:_batch_size], self.device)
      
        loss, loss_pm = self.segmentator.train(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                nd_target_word_idx, nd_target_valid_len, nd_target_segment,
                                nd_pm_error_idx, nd_pm_add_idx, nd_pm_remove_idx, 
                                list_input_texts[:_batch_size], list_target_texts[:_batch_size], self.device, _batch_size, self.trainer)
                                
        if (_step % 100 == 0):
        
          print('Review Loss => {}, Epoch => {}, Lr => {}'.format(loss, self.review_epoch, self.trainer.learning_rate))
          self.sw.add_scalar(tag = 'review_loss', value = loss, global_step = _step)
          if self.config['use_encoder_constraint']:
            self.sw.add_scalar(tag = 'pm_loss', value = loss_pm, global_step = _step)
        
        progress_review.update(_batch_size)
        
      elif _dataset == 'csc15':
        if not self.config['csc_fixed']:
          i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, list_input_texts, list_target_texts) = next(self.csc14_enumerator, (-1, [None] * 8))
        else:
          i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, list_input_texts, list_target_texts) = next(self.csc14_enumerator, (-1, [None] * 6))
        if i == -1:
          self.csc15loader = self.csc15data.get_loader()
          self.csc15_epoch += 1
          self.csc15_enumerator = enumerate(self.csc15loader)
          i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, list_input_texts, list_target_texts) = next(self.csc15_enumerator, (-1, [None] * 8))
          progress_csc15 = tqdm(total = self.int_csc15_samples, desc = 'csc15')
          
        _batch_size = int(nd_input_word_idx.shape[0] / len(self.device)) * len(self.device)
        nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx[:_batch_size], self.device)
        nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len[:_batch_size], self.device)
        nd_input_segment = gluon.utils.split_and_load(nd_input_segment[:_batch_size], self.device)
        
        nd_target_word_idx = gluon.utils.split_and_load(nd_target_word_idx[:_batch_size], self.device)
        # nd_target_valid_len = gluon.utils.split_and_load(nd_target_valid_len[:_batch_size], self.device)
        # nd_target_segment = gluon.utils.split_and_load(nd_target_segment[:_batch_size], self.device)
        if not self.config['csc_fixed']:
          nd_target_valid_len = gluon.utils.split_and_load(nd_target_valid_len[:_batch_size], self.device)
          nd_target_segment = gluon.utils.split_and_load(nd_target_segment[:_batch_size], self.device)
      
          loss = self.segmentator.train(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                  nd_target_word_idx, nd_target_valid_len, nd_target_segment,
                                  list_input_texts[:_batch_size], list_target_texts[:_batch_size], self.device, _batch_size, self.trainer)
                                    
        else:
          loss = self.segmentator.train_csc_fixed(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                    nd_target_word_idx, self.device, _batch_size, self.trainer)
        
        
                                
        # loss = self.segmentator.train(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
        #                        nd_target_word_idx, nd_target_valid_len, nd_target_segment,
        #list_input_texts[:_batch_size], list_target_texts[:_batch_size], self.device, _batch_size, self.trainer)
                                
        if (_step % 100 == 0):
        
          print('CSC15 Loss => {}, Epoch => {}, Lr => {}'.format(loss, self.csc15_epoch, self.trainer.learning_rate))
        
        progress_csc15.update(_batch_size)
      #   print('Epoch {} Step {} => {}, LR : {}'.format(e, i, loss, self.trainer.learning_rate))
      elif _dataset == 'csc14':
        
          if not self.config['csc_fixed']:
            i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, list_input_texts, list_target_texts) = next(self.csc14_enumerator, (-1, [None] * 8))
          else:
            i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, list_input_texts, list_target_texts) = next(self.csc14_enumerator, (-1, [None] * 6))
          
          if i == -1:
            self.csc14loader = self.csc14data.get_loader()
            self.csc14_epoch += 1
            self.csc14_enumerator = enumerate(self.csc14loader)
            i, (nd_input_word_idx, nd_input_valid_len, nd_input_segment, nd_target_word_idx, nd_target_valid_len, nd_target_segment, list_input_texts, list_target_texts) = next(self.csc14_enumerator, (-1, [None] * 8))
            progress_csc14 = tqdm(total = self.int_csc14_samples, desc = 'csc14')
            
          _batch_size = int(nd_input_word_idx.shape[0] / len(self.device)) * len(self.device)
          nd_input_word_idx = gluon.utils.split_and_load(nd_input_word_idx[:_batch_size], self.device)
          nd_input_valid_len = gluon.utils.split_and_load(nd_input_valid_len[:_batch_size], self.device)
          nd_input_segment = gluon.utils.split_and_load(nd_input_segment[:_batch_size], self.device)
          
          nd_target_word_idx = gluon.utils.split_and_load(nd_target_word_idx[:_batch_size], self.device)
          
          if not self.config['csc_fixed']:
            nd_target_valid_len = gluon.utils.split_and_load(nd_target_valid_len[:_batch_size], self.device)
            nd_target_segment = gluon.utils.split_and_load(nd_target_segment[:_batch_size], self.device)
        
            loss = self.segmentator.train(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                    nd_target_word_idx, nd_target_valid_len, nd_target_segment,
                                    list_input_texts[:_batch_size], list_target_texts[:_batch_size], self.device, _batch_size, self.trainer)
                                    
          else:
            loss = self.segmentator.train_csc_fixed(nd_input_word_idx, nd_input_valid_len, nd_input_segment,
                                    nd_target_word_idx, self.device, _batch_size, self.trainer)
          
          
                                  
          if (_step % 100 == 0):
          
            print('CSC14 Loss => {}, Epoch => {}, Lr => {}'.format(loss, self.csc14_epoch, self.trainer.learning_rate))
          
          progress_csc14.update(_batch_size)
      #   print('Epoch {} Step {} => {}, LR : {}'.format(e, i, loss, self.trainer.learning_rate))
      # except Exception as e:
      #   # print(e)
      #   continue
      # if e % self.save_freq == 0:
      #   save_gluon_model(self.segmentator, self.arch_path, e, 0) # use main     dataset
      if _step % self.save_freq_step == 0:
        save_gluon_model(self.segmentator, self.arch_path, _step, 0) # use main     dataset
        
      if _step % self.config['val_freq'] == 0 and _step != 0: 
        # _, batch_test = self.review_enumerator_val.__next__()
        # list_input_texts_test = batch_test[0]
        # list_target_texts_test = batch_test[1]
        
        # print('Input => ', list_input_texts_test)
        # print('Target => ', list_target_texts_test)
        
        # text = self.segmentator.run(list_input_texts_test, self.device)
        
        avg_val_bleu, predict_text = self.val_using_testset()
      
        self.sw.add_scalar(tag = 'avg_val_bleu', value = avg_val_bleu, global_step = _step)
      
      # self.options['learning_rate'] = self.trainer.learning_rate * self.lr_decay_rate_epoch
      
      # self.trainer = mx.gluon.Trainer(self.segmentator.collect_params(), self.optimizer, self.options)

  def test(self):
  
    self.segmentator.hybridize()
    self.loader = self.reviewdata_val.get_loader()
    for i, batch in enumerate(self.loader):      
     
      str_input_text, str_target_text = batch
      str_predict_text = self.segmentator.run(str_input_text, self.test_device)
      
      score = sentence_bleu([[t for t in str_predict_text]], [t for t in str_target_text])
      raise
    
    pass
    
  def val_using_testset(self):
  
    # self.segmentator.hybridize()
    progress_val = tqdm(total = self.config['int_val_set'], desc = 'cged')

    self.loader = self.reviewdata_val.get_loader()
    scores = []
    for i, batch in enumerate(self.loader):
      if i == self.config['int_val_set']:
        break
      str_input_text, str_target_text = batch
      str_predict_text = self.segmentator.run(str_input_text, self.test_device)
      score = sentence_bleu([[t for t in str_predict_text]], [t for t in str_target_text])
      scores.append(score)
      progress_val.update(1)
    
    return np.mean(scores), str_predict_text
    
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
    self.loader = self.cgeddata.get_loader()
    self.num_samples = len(self.cgeddata.data)
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
        
      _list_result = self.cgeddata.convert_to_report_format(list_ids, _predict_error_idx)
  
      list_result.extend(_list_result)
      
      progress.update(self.batch_size)
      
    self.cgeddata.gen_eval_report(list_result)
    
  def train_paper(self):
  
    epoch_start, _ = self.init_network()
    epoch_start += 1
    
    self.init_trainer(epoch_start)
    self.loader = self.data.get_loader()
    # self.loader_val = self.data_val.get_loader()
    
    self.num_samples = len(self.data.data)
    self.segmentator.hybridize()
    pass
  
  def test_CSC(self):
  
    self.init_network()
    self.segmentator.hybridize()
    if self.args.dataset == 'CSC14':
      self.cscdata = self.csc14data
      self.cscloader = self.csc14data.get_loader()
      self.int_samples = len(self.csc14data.data)
    else:
      self.cscloader = self.csc15data.get_loader()
      self.int_samples = len(self.csc15data.data)
      self.cscdata = self.csc15data
    
    self.num_samples = len(self.cscdata.data)

    progress = tqdm(total = self.num_samples)
      
    
    reports = []  
    for i, batch in enumerate(self.cscloader):
      nd_input_word_idx, nd_input_valid_len, nd_input_segment,  list_input_texts, list_target_text, list_ids = batch
      print('input => ', list_input_texts[0])
      print('target => ', list_target_text[0])
      prediction = self.segmentator.run(list_input_texts[0], self.device)
      
      report = self.cscdata.compare_input_prediction(list_ids[0], prediction,  list_input_texts[0])
      reports.append(report)
      progress.update(1)
      
    self.csc15data.gen_eval_report(reports)
      
      
      

    

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
    elif 'CSC' in args.dataset:
      s.test_CSC()
    else:
      s.test()
    
  elif args.run:
  
    s.run(args.text)
    
  elif args.interact:
  
    s.interact()
