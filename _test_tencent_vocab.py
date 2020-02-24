from utils import load_pickle, save_pickle
from opencc import OpenCC                
s2t = OpenCC('s2t')  # 
t2s = OpenCC('t2s') 

p = load_pickle('./tencent/vocab')

tc = []

for _p in p:

  tc.append(s2t.convert(_p))
  
save_pickle(tc, "tencent_vocab_tc")
  
print(tc)