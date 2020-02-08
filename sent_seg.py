import gluonnlp as nlp; import mxnet as mx;
import gluonnlp.model.transformer as trans
from opencc import OpenCC                                                                          
cc = OpenCC('s2t')  # 
# model, vocab
#= nlp.model.get_model('bert_12_768_12', dataset_name = 'wiki_multilingual_uncased', use_classifier = False, use_decoder = False);
mode, vocab = nlp.model.bert_12_768_12(dataset_name='wiki_multilingual_uncased', pretrained=False)
tokenizer = nlp.data.BERTTokenizer(vocab, lower = True);
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length = 512, pair = False, pad = False);


tokens = tokenizer('gluonnlp: 使NLP变得简单。')

print(tokens)

raise

sample = transform(('gluonnlp: 使NLP变得简单。',))
print(sample)

for t in tokens:
  print(t[0])
  print(vocab.token_to_idx[t])

idx = [vocab.token_to_idx[t] for t in tokens]

print(idx)

raise

print(vocab.token_to_idx)

input_ids = tokenizer.convert_tokens_to_ids(tokens)

print(input_ids)


words, valid_len, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]])

seq_encoding, cls_encoding = model(words, segments, valid_len);

decoder.initialize()

decoder_state = decoder.init_state_from_encoder(seq_encoding, valid_len)

output = decoder(seq_encoding, decoder_state)


decode_output, _, _ = self.reconstructor.decode_seq(batch_major_text[:, : max_td -1], reconstructor_state, td_nd)                                                                           

temp_outputs[i] = nd.softmax(self.fc(self.ln_fc(self.dropout(decode_output)))).swapaxes(0, 1)


