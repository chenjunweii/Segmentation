import cmd
import os
from mxnet_utils.model import load_latest_checkpoint, save_gluon_model, load_pretrained_model, load_pretrained_model_only_same_shape, load_pretrained_model_only_same_shape_sup  

class shell(cmd.Cmd):
  def __init__(self, segmentator, arch_path, device):
    super(shell, self).__init__()
    self.prompt = ">> "
    self.segmentator = segmentator
    self.arch_path = arch_path
    self.device = device
    
  def do_load(self, epoch):
    # print(os.path.join('model', self.arch_path, '{:04d}-0.params'.format(int(epoch))))
    load_pretrained_model(self.segmentator, os.path.join(self.arch_path, '{:04d}-0.params'.format(int(epoch))), self.device)
  def default(self, text):
    if text == '':
      return
    # print('[*] Possible Result for {} : '.format(text))
    self.segmentator.run(text, self.device)