import mxnet as mx
import numpy as np
from mxnet import gluon, nd, autograd
from mxnet.gluon import Block, nn
import gluonnlp as nlp
import gluonnlp.model.transformer as trans
from random import choice
from opencc import OpenCC
from zhon import cedict, hanzi, zhuyin
import string
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as sce
s2t = OpenCC('s2t')  # 
t22 = OpenCC('t2s') 


class Segmentator(Block):

  def __init__(self, args, config):
  
    super(Segmentator, self).__init__()
    
    # self.actions = actions
    self.pms = config['list_puncuation_marks']
    self.config = config
    self.num_layers = 12
    self.num_heads = 12
    self.hidden_size = 512
    self.max_seq_length = config['int_max_length']
    self.units = 768
    self.args = args
    
    with self.name_scope():
      self.encoder, self.vocab_src = nlp.model.get_model('bert_12_768_12', dataset_name = 'wiki_cn_cased', use_classifier = False, use_decoder = False, pretrained = True);
      
      # if (self.args.use_tc):
      self.counter_tgt = nlp.data.count_tokens(self.config['str_character_target'])
      self.vocab_tgt = nlp.vocab.BERTVocab(self.counter_tgt)
    
      self.dropout = nn.Dropout(0.5) 
      self.decoder = trans.TransformerDecoder(attention_cell = 'multi_head', 
                                              num_layers = self.num_layers,
                                              units = self.units, hidden_size = self.hidden_size, max_length = self.max_seq_length,
                                              num_heads = self.num_heads, scaled=True, dropout=0.1,
                                              use_residual = True, output_attention=False,
                                              weight_initializer=None, bias_initializer='zeros',
                                              scale_embed=True, prefix=None, params=None)
      
      # self.decoder_action = trans.TransformerDecoder(attention_cell = 'multi_head', 
      #                                         num_layers = self.num_layers,
      #                                         units = self.units, hidden_size = self.hidden_size, max_length = self.max_seq_length,
      #                                         num_heads = self.num_heads, scaled=True, dropout=0.1,
      #                                         use_residual = True, output_attention=False,
      #                                         weight_initializer=None, bias_initializer='zeros',
      #                                         scale_embed=True, prefix=None, params=None)
                                              
      # self.fc_actions = nn.Dense(len(self.actions), flatten = False)
      # self.fc_pms = nn.Dense(len(self.pms), flatten = False)
      self.fc_proj = nn.Dense(len(self.vocab_tgt), flatten = False)
      self.emb_tgt = nn.HybridSequential()
      
      self.emb_tgt.add(nn.Embedding(len(self.vocab_tgt), self.units))
      self.emb_tgt.add(nn.Dropout(0.5))
      
      # self.emb_actions = (nn.Embedding(input_dim = len(self.actions), output_dim = self.units))
      # self.emb_pms = (nn.Embedding(input_dim = len(self.pms), output_dim = self.units))
      # self.emb 
      self.tokenizer = nlp.data.BERTTokenizer(self.vocab_src, lower = False);
      self.transformer = nlp.data.BERTSentenceTransform(self.tokenizer, max_seq_length = self.max_seq_length, pair = False, pad = True);
    self.beam_scorer = nlp.model.BeamSearchScorer()
    self.beam_sampler = nlp.model.BeamSearchSampler(beam_size = 5,
                                           decoder = self._decode_step,
                                           eos_id = self.vocab_tgt.token_to_idx[self.vocab_tgt.sep_token],
                                           scorer = self.beam_scorer,
                                           max_length = self.max_seq_length)
      
  def deprecated__decode(self, inputs_text, predict_action, predict_pm):
  
    predict_text = ''
    
    predict_action = predict_action.argmax(-1)
    
    predict_pm = predict_pm.argmax(-1)
      
    inputs_token = self.tokenizer(inputs_text)
    
    _idx = 0
    
    for _idx, _input_text in enumerate(inputs_token):
    
      action = self.actions[int(predict_action[_idx].asnumpy())]
    
      pm = self.pms[int(predict_pm[_idx].asnumpy())]
      
      if (action == 'add') and pm is not None:
      
        predict_text = predict_text + pm + _input_text
        
      # elif (action == None):
      
      else:
      
        predict_text = predict_text + _input_text
        
    # predict_text = predict_text + self.pms[int(predict_pm[_idx + 1].asnumpy())] # 句號
    

        
    return predict_text
    
    
    # for act, pm in zip(predict_action, predict_pm):
    
  def balance_multi_objective(self, action_idx, pm_idx, target_action_idx, target_pm_idx, ratio = 1):
  
    # 以 target 的 idx 爲基準
    max_valid_len = action_idx.shape[1] # input valid length
    action_None = (action_idx == 0)
    action_target_None = (target_action_idx == 0)[:, 1:max_valid_len + 1]
    num_action_None = action_None.sum()
    num_action_target_None = action_target_None.sum()
    pm_None = (pm_idx == 0)
    pm_target_None = (target_pm_idx == 0)[:, 1:max_valid_len + 1]
    num_pm_None = pm_None.sum()
    num_pm_target_None = num_pm_None.sum()
    action_Else = (action_idx != 0)
    action_target_Else = (target_action_idx != 0)[:, 1:max_valid_len + 1]
    num_action_Else = action_Else.sum()
    num_action_target_Else = action_target_Else.sum()
    pm_Else = (pm_idx != 0)
    pm_target_Else = (target_pm_idx != 0)[:, 1:max_valid_len + 1]
    num_pm_Else = pm_Else.sum()
    num_pm_target_Else = pm_target_Else.sum()
    
    if (num_action_None > num_action_Else):
      action_None = action_None.reshape(-1).topk(k = min(int(ratio * num_action_Else.asnumpy()), int(num_action_None.asnumpy())), ret_typ = 'mask').reshape(action_None.shape)
      num_action_None = action_None.sum()
      
    # else:
    #   action_Else = action_Else.reshape(-1).topk(k = min(int(ratio * num_action_None.asnumpy()), int(num_action_Else.asnumpy())), ret_typ = 'mask').reshape(action_Else.shape)
    #   num_action_Else = action_Else.sum()
    
    if (num_pm_None > num_pm_Else):
      pm_None = pm_None.reshape(-1).topk(k = min(int(ratio * num_pm_Else.asnumpy()), int(num_pm_None.asnumpy())), ret_typ = 'mask').reshape(pm_None.shape)
      num_pm_None = pm_None.sum()
      
    
    action_target_None = action_target_None.reshape(-1).topk(k = min(int(ratio * num_action_target_Else.asnumpy()), int(num_action_target_None.asnumpy())), ret_typ = 'mask').reshape(action_target_None.shape)
    pm_target_None = pm_target_None.reshape(-1).topk(k = min(int(ratio * num_pm_target_Else.asnumpy()), int(num_pm_target_None.asnumpy())), ret_typ = 'mask').reshape(pm_target_None.shape)
    num_action_target_None = action_target_None.sum()
    num_pm_target_None = pm_target_None.sum()


    #action_mask = (action_None + action_Else + (action_target_None + action_target_Else)).clip(0, 1)
    #pm_mask = (pm_None + pm_Else + (pm_target_None + pm_target_Else)).clip(0, 1)
    
    action_mask = (action_target_None + action_target_Else).clip(0, 1)
    
    pm_mask = (pm_target_None + pm_target_Else).clip(0, 1)
    
    # print('action_mask : ', action_mask.sum())
    
    # print('num_action_None : ', num_action_None)
    
    # print('num_action_Else : ', num_action_Else)
    
    # print('action_target_else : ', num_action_target_Else)
    
    # print('action_Target_none : ', num_action_target_None)
    # print('pm_mask : ', pm_mask.sum())
      
    return action_mask.expand_dims(-1).detach(), pm_mask.expand_dims(-1).detach()
    
  def ce(self, p, g):                                                                                                                                                                              
    p_clip = nd.clip(p, 1e-20, 1)
    _p_clip = nd.clip(1 - p, 1e-20, 1)
    return (- g * nd.log(p_clip)).sum(axis = -1)
    
  def decode_greedy(self, predict_output_logit):
    predict_output_idx = predict_output_logit.argmax(-1).asnumpy()
    return ''.join([self.vocab_tgt.idx_to_token[int(idx)] for idx in predict_output_idx])
    
  def decode_beamsearch(self, decoder_state, batch_size, device):
  
    decode = []
  
    start_idx = mx.nd.full(shape = (batch_size), ctx = device, dtype = np.float32,
                            val = self.vocab_tgt.token_to_idx[self.vocab_tgt.cls_token])
    
    sample, score, valid_len = self.beam_sampler(start_idx, decoder_state)
    
    
    
    for beam, _score, _len in zip(sample[0].asnumpy(), score[0].asnumpy(), valid_len[0].asnumpy()):
    
      decode.append((''.join([ self.vocab_tgt.idx_to_token[_beam] for _beam in beam[:_len]])).replace('[PAD]', '') + ', score : {}'.format(_score))
      
    return decode
  
  def _decode_step(self, step_input, state):
    step_output, state, _ = self.decoder(self.encoder.word_embed(step_input), state)
    step_output = self.fc_proj(step_output)
    return nd.log_softmax(step_output), state
    
  def train(self, input_word_idx, input_len, input_seg, target_word_idx, target_len, target_seg, inputs_text, targets_text, devices, batch_size, trainer):
  
    seq_encoding = [None] * len(devices); cls_encoding = [None] * len(devices)
    decoder_state = [None] * len(devices); target_word_emb = [None] * len(devices)
    predict_word_emb = [None] * len(devices); predict_word_logit = [None] * len(devices)
    target_word_logit = [None] * len(devices); input_word_logit = [None] * len(devices)
    loss = [None] * len(devices)
    num_device = len(devices)
  
    for i in range(num_device):
      seq_encoding[i], cls_encoding[i] = self.encoder(input_word_idx[i], input_seg[i], input_len[i])
    
    # nd.waitall()
    for i in range(num_device):
      with autograd.record():
    
      #""" Decoder with word"            
        # seq_encoding[i], cls_encoding[i] = self.encoder(input_word_idx[i], input_seg[i], input_len[i])
        decoder_state[i] = self.decoder.init_state_from_encoder(seq_encoding[i], input_len[i])
        target_word_emb[i] = self.emb_tgt(target_word_idx[i])
        predict_word_emb[i], _, _ = self.decoder.decode_seq(target_word_emb[i], decoder_state[i])#, valid_len)
        
        
        # target_word_logit_train = nd.softmax(self.fc_proj(target_word_emb[i]))
        
        # print(target_word_logit_train.shape)
        
        # print(target_word_logit[i].shape)
        
        # raise
        
        predict_word_logit[i] = nd.softmax(self.fc_proj(predict_word_emb[i]))
        target_word_logit[i] = nd.one_hot(target_word_idx[i], len(self.vocab_tgt))
        input_word_logit[i] = nd.one_hot(input_word_idx[i], len(self.vocab_src))
        
        max_target_len = int(max(target_len[i].asnumpy()))
        loss[i] = self.ce(predict_word_logit[i][:, : max_target_len - 1], target_word_logit[i][:, 1 : max_target_len])
        
        #loss[i] = loss[i].mean([1]) + (((predict_word_emb[i][:, : max_target_len - 1]) - target_word_emb[i][:, 1 : max_target_len]) ** 2).mean([1, 2])
        
        #+ self.ce(target_word_logit_train[:, 1 : max_target_len], target_word_logit[i][:, 1 : max_target_len])


      # targets_action_embs = self.emb_actions(targets_action)
      # targets_pm_embs = self.emb_pms(targets_pm)
      
      # max_valid_len = int(valid_len.max().asnumpy())
      
      # action_output_embs, _, _ = self.decoder_action.decode_seq(targets_action_embs[ : , : max_valid_len], decoder_action_state)#, valid_len)
      
      # """ Decoder """
      # decoder_pm_state = self.decoder_pm.init_state_from_encoder(seq_encoding, valid_len)
      # decoder_action_state = self.decoder_action.init_state_from_encoder(seq_encoding, valid_len)
      
      # targets_action_embs = self.emb_actions(targets_action)
      # targets_pm_embs = self.emb_pms(targets_pm)
      
      # max_valid_len = int(valid_len.max().asnumpy())
      
      # action_output_embs, _, _ = self.decoder_action.decode_seq(targets_action_embs[ : , : max_valid_len], decoder_action_state)#, valid_len)
      # pm_output_embs, _, _ = self.decoder_pm.decode_seq(targets_pm_embs[ : , : max_valid_len], decoder_pm_state)#, valid_len)                                                                        
      
      # action_output = nd.softmax(self.fc_actions(self.dropout(action_output_embs)))
      # pm_output = nd.softmax(self.fc_pms(self.dropout(pm_output_embs)))
      
      # action_idx = action_output.argmax(-1)
      # pm_idx = pm_output.argmax(-1)
      
      # action_mask, pm_mask = self.balance_multi_objective(action_idx, pm_idx, targets_action, targets_pm, 3)
      
      # targets_action_logits = nd.one_hot(targets_action, len(self.actions))
      # targets_pm_logits = nd.one_hot(targets_pm, len(self.pms))
      
      # action_loss = self.ce(action_output  * action_mask, targets_action_logits[:, 1 : max_valid_len + 1] * action_mask)
      # pm_loss = self.ce(pm_output * pm_mask, targets_pm_logits[:,1 : max_valid_len + 1] * pm_mask)
      
      # loss = action_loss / action_mask.sum().detach() + pm_loss / pm_mask.sum().detach()
      
      
      # """ Decoder End """
      
      # """ Encoder Start """
      
      
      # targets_action_logits = nd.one_hot(targets_action, len(self.actions))
      # targets_pm_logits = nd.one_hot(targets_pm, len(self.pms))
      
      # action_output = nd.softmax(self.fc_actions(self.dropout(seq_encoding)))
      # pm_output = nd.softmax(self.fc_pms(self.dropout(seq_encoding)))
      
      # max_valid_len = int(valid_len.max().asnumpy())
      
      # action_idx = action_output.argmax(-1)
      # pm_idx = pm_output.argmax(-1)
      
      # action_mask, pm_mask = self.balance_multi_objective(action_idx, pm_idx, targets_action, targets_pm, 3)
      
      # action_loss = self.ce(action_output[:, :max_valid_len ] * action_mask[:, :max_valid_len],
      #               targets_action_logits[:, :max_valid_len] * action_mask[:, :max_valid_len])
                    
      # pm_loss = self.ce(pm_output[:, :max_valid_len ] * pm_mask[:, :max_valid_len],
      # targets_pm_logits[:, :max_valid_len] * pm_mask[:, :max_valid_len])
      
      # loss = action_loss.sum() / action_mask.sum() + pm_loss.sum() / pm_mask.sum()
      
      # """ Encoder End """
      
    # debug_action_loss = self.ce((action_output  * action_mask) [0:,  : max_valid_len], targets_action_logits[:, 1 : max_valid_len + 1] * action_mask)
    # debug_pm_loss = self.ce(pm_output[:, : max_valid_len] * pm_mask, targets_pm_logits[0:,1:max_valid_len + 1] * pm_mask)
      
    # print('action loss : ', (action_loss / action_mask.sum()).sum())
    # print('pm_loss : ', (pm_loss / pm_mask.sum()).sum())
    # nd.waitall()

    for _loss in loss:  
      _loss.backward()
    # nd.waitall()

    
    nd.waitall()
    # decode_text = self.decode(inputs_text[0], action_output[0], pm_output[0])
    
    # decode_text_debug = self.decode(inputs_text[0], targets_action_logits[0, 1 : ], targets_pm_logits[0, 1:])
    
    # print('debug => ', decode_text_debug)
    #self.decode_beamsearch(decoder_state[0], int(batch_size / len(devices)), devices[0])
    
    trainer.step(batch_size, ignore_stale_grad = True)
    
    loss = sum([_loss.mean().asnumpy() for _loss in loss])
    
    return loss, self.decode_greedy(predict_word_logit[0][0]).replace('[PAD]', '')
      
      