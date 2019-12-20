import functools
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from itertools import zip_longest

from common.subword.sentence_encoder import SentenceEncoder
from common.bleu.compute_bleu_by_nist_format import compute_bleu_score

import warnings
warnings.filterwarnings('ignore')

'''
Training Attention-based Neural Machine Translation Model
'''

# data
src_max_len = 50
trg_max_len = 50
train_src = 'corpus/train.cn'
train_trg = 'corpus/train.en'
valid_src = 'corpus/train.cn'
valid_trg = ['corpus/train.en']
nist_src = 'corpus/vali.src'
nist_ref = 'corpus/vali.ref'
vfreq = 5  # frequency for validation

# model
model_name = 'RNNSearch'
checkpoint_name = ''
enc_ninp = 256  # size of source word embedding
dec_ninp = 256  # size of target word embedding
enc_nhid = 256  # units of source hidden layer
dec_nhid = 256  # units of target hidden layer
dec_natt = 256  # units of target attention layer
nreadout = 256  # units of maxout layer
enc_emb_dropout = 0.3  # dropout rate for encoder embedding
dec_emb_dropout = 0.3  # dropout rate for decoder embedding
enc_hid_dropout = 0.3  # dropout rate for encoder hidden state
readout_dropout = 0.3  # dropout rate for readout layer
nepoch = 1000  # number of epochs to train

# optimization
optim_name = 'RMSprop'
batch_size = 2
lr = 0.0005
l2 = 0  # L2 regularization
grad_clip = 1  # gradient clipping
decay_lr = False  # decay learning rate
# half_epoch = False  # decay learning rate at the beginning of epoch
restore = False  # decay learning rate at the beginning of epoch
cuda = False
local_rank = None

# bookkeeping
seed = 123  # random number seed
checkpoint = './checkpoint/'  # path to save the model



cuda = cuda and torch.cuda.is_available()
device_type= 'cuda' if cuda else 'cpu'
device_ids = None
if local_rank is not None:
    device_type += ':' + str(local_rank)
    device_ids = [local_rank]
device = torch.device(device_type)

# load vocabulary for source and target
encoder_src = SentenceEncoder('corpus/vob.src')
encoder_trg = SentenceEncoder('corpus/vob.tgt')
PAD = 0
UNK = 1
SOS = 2
EOS = 3
enc_ntok = 30000
dec_ntok = 30000


# load dataset for training and validation
class dataset(torch.utils.data.Dataset):
  def __init__(self, p_src, p_trg, src_max_len=None, trg_max_len=None):
    p_list = [p_src]
    if isinstance(p_trg, str):
      p_list.append(p_trg)
    else:
      p_list.extend(p_trg)
    lines = []
    for p in p_list:
      with open(p) as f:
        lines.append(f.readlines())
    assert len(lines[0]) == len(lines[1])
    self.data = []
    for line in zip_longest(*lines):
      line = [v.lower().strip() for v in line]
      if not any(line):
        continue
      line = [v.split() for v in line]

      # if (src_max_len and len(line[0]) > src_max_len) \
      #     or (trg_max_len and len(line[1]) > trg_max_len):
      #   continue
      self.data.append(line)
    self.length = len(self.data)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    return self.data[index]


train_dataset = dataset(train_src, train_trg, src_max_len, trg_max_len)
valid_dataset = dataset(valid_src, valid_trg)
train_iter = torch.utils.data.DataLoader(
  train_dataset, batch_size, shuffle=False, num_workers=4,
  collate_fn=lambda x: zip(*x)
)
valid_iter = torch.utils.data.DataLoader(
  valid_dataset, 1, shuffle=False, collate_fn=lambda x: zip(*x)
)


# model part
class Encoder(nn.Module):
  """"encode the input sequence with Bi-GRU"""
  def __init__(self, ninp, nhid, ntok, padding_idx, emb_dropout, hid_dropout):
    super(Encoder, self).__init__()
    self.nhid = nhid
    self.emb = nn.Embedding(ntok, ninp, padding_idx=padding_idx)
    self.bi_gru = nn.GRU(ninp, nhid, 1, batch_first=True, bidirectional=True)
    self.gru_1 = nn.GRU(2 * nhid, nhid, 1, batch_first=True, bidirectional=False)
    self.gru_2 = nn.GRU(nhid, nhid, 2, batch_first=True, bidirectional=False)
    self.enc_emb_dp = nn.Dropout(emb_dropout)
    self.enc_hid_dp = nn.Dropout(hid_dropout)

  def init_hidden(self, batch_size, layer_num):
    weight = next(self.parameters())
    h0 = weight.new_zeros(layer_num, batch_size, self.nhid)
    return h0

  def forward(self, input, mask):
    hidden = self.init_hidden(input.size(0), 2)
    h_1 = self.init_hidden(input.size(0), 1)
    h_2 = self.init_hidden(input.size(0), 2)
    # self.bi_gru.flatten_parameters()
    input = self.enc_emb_dp(self.emb(input))
    # true sentence length, a vector denoting every sentence length in one batch
    length = mask.sum(1).tolist()
    # length after padding
    total_length = mask.size(1)
    input = torch.nn.utils.rnn.pack_padded_sequence(input, length,
                                                    batch_first=True)
    output, hidden = self.bi_gru(input, hidden)
    output, _ = self.gru_1(output, h_1)
    output, _ = self.gru_2(output, h_2)
    output = torch.nn.utils.rnn.pad_packed_sequence(
      output, batch_first=True,
      total_length=total_length)[0]
    output = self.enc_hid_dp(output)
    # hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
    return output, hidden


class Attention(nn.Module):
  """Attention Mechanism"""
  def __init__(self, nhid, ncontext, natt):
    super(Attention, self).__init__()
    self.h2s = nn.Linear(nhid, natt)
    self.s2s = nn.Linear(ncontext, natt)
    self.a2o = nn.Linear(natt, 1)

  def forward(self, hidden, mask, context):
    shape = context.size()
    attn_h = self.s2s(context.view(-1, shape[2]))
    attn_h = attn_h.view(shape[0], shape[1], -1)
    attn_h += self.h2s(hidden).unsqueeze(1).expand_as(attn_h)
    logit = self.a2o(F.tanh(attn_h)).view(shape[0], shape[1])
    if mask.any():
      logit.data.masked_fill_(1 - mask, -float('inf'))
    softmax = F.softmax(logit, dim=1)
    output = torch.bmm(softmax.unsqueeze(1), context).squeeze(1)
    return output


class VallinaDecoder(nn.Module):
  def __init__(self, ninp, nhid, enc_ncontext, natt, nreadout, readout_dropout):
    super(VallinaDecoder, self).__init__()
    self.gru1 = nn.GRUCell(ninp, nhid)
    self.gru2 = nn.GRUCell(enc_ncontext, nhid)
    self.enc_attn = Attention(nhid, enc_ncontext, natt)
    self.e2o = nn.Linear(ninp, nreadout)
    self.h2o = nn.Linear(nhid, nreadout)
    self.c2o = nn.Linear(enc_ncontext, nreadout)
    self.readout_dp = nn.Dropout(readout_dropout)

  def forward(self, emb, hidden, enc_mask, enc_context):
    hidden = self.gru1(emb, hidden)
    attn_enc = self.enc_attn(hidden, enc_mask, enc_context)
    hidden = self.gru2(attn_enc, hidden)
    output = F.tanh(self.e2o(emb) + self.h2o(hidden) + self.c2o(attn_enc))
    output = self.readout_dp(output)
    return output, hidden


class RNNSearch(nn.Module):
  def __init__(self, enc_nhid, dec_nhid,
               enc_ntok, dec_ntok, dec_ninp, enc_ninp,
               enc_emb_dropout, enc_hid_dropout,
               dec_emb_dropout,
               dec_natt,
               nreadout, readout_dropout):
    super(RNNSearch, self).__init__()
    self.dec_nhid = dec_nhid

    self.emb = nn.Embedding(dec_ntok, dec_ninp, padding_idx=PAD)
    self.encoder = Encoder(enc_ninp, enc_nhid, enc_ntok,
                           PAD, enc_emb_dropout,
                           enc_hid_dropout)
    # self.decoder = VallinaDecoder(dec_ninp, dec_nhid, 2 * enc_nhid,
    #                               dec_natt, nreadout,
    #                               readout_dropout)
    self.decoder = VallinaDecoder(dec_ninp, dec_nhid, enc_nhid,
                                  dec_natt, nreadout,
                                  readout_dropout)
    self.affine = nn.Linear(nreadout, dec_ntok)
    # self.init_affine = nn.Linear(2 * enc_nhid, dec_nhid)
    self.init_affine = nn.Linear(enc_nhid, dec_nhid)
    self.dec_emb_dp = nn.Dropout(dec_emb_dropout)

  def forward(self, src, src_mask, f_trg, f_trg_mask, b_trg=None,
              b_trg_mask=None):
    enc_context, _ = self.encoder(src, src_mask)
    enc_context = enc_context.contiguous()

    avg_enc_context = enc_context.sum(1)
    enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)
    avg_enc_context = avg_enc_context / enc_context_len  # avg every sentence

    attn_mask = src_mask.byte()

    hidden = F.tanh(self.init_affine(avg_enc_context))

    loss = 0
    for i in range(f_trg.size(1) - 1):
      output, hidden = self.decoder(self.dec_emb_dp(self.emb(f_trg[:, i])),
                                    hidden, attn_mask, enc_context)
      loss += F.cross_entropy(self.affine(output), f_trg[:, i + 1],
                              reduce=False) * f_trg_mask[:, i + 1]
    w_loss = loss.sum() / f_trg_mask[:, 1:].sum()
    loss = loss.mean()
    return loss.unsqueeze(0), w_loss.unsqueeze(0)

  def predict(self, src, src_mask, max_len=None):
    # src.size(0): 1
    max_len = src.size(1) * 3 if max_len is None else max_len

    enc_context, _ = self.encoder(src, src_mask)
    enc_context = enc_context.contiguous()

    avg_enc_context = enc_context.sum(1)
    enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)
    avg_enc_context = avg_enc_context / enc_context_len

    attn_mask = src_mask.byte()

    hidden = F.tanh(self.init_affine(avg_enc_context))

    hyp = [2]
    input = torch.tensor(hyp) # src[:, 0]: tensor([3])
    for k in range(max_len):
      input = self.dec_emb_dp(self.emb(input))
      output, hidden = self.decoder(input, hidden, attn_mask, enc_context)
      log_prob = F.log_softmax(self.affine(output), dim=1)
      # log_prob.size(0): 1, log_prob.size(1): 30000
      pred_tensor = log_prob.argmax(dim=1) # shape should be (batch_size, 1)
      pred_id = pred_tensor.tolist()[0]
      if pred_id == 3:
        break
      else:
        input = pred_tensor
        hyp.append(pred_id)

    return hyp


# create the model
model = RNNSearch(enc_nhid, dec_nhid,
                  enc_ntok, dec_ntok, dec_ninp, enc_ninp,
                  enc_emb_dropout, enc_hid_dropout,
                  dec_emb_dropout,
                  dec_natt,
                  nreadout, readout_dropout).to(device)

# initialize the parameters
for p in model.parameters():
  p.data.uniform_(-0.1, 0.1)
param_list = list(model.parameters())
param_group = param_list

# create the optimizer
optimizer = getattr(optim, optim_name)(param_group, lr=lr, weight_decay=l2)

score_list = []
cur_lr = ' '.join(map(lambda g: str(g['lr']), optimizer.param_groups))
best_name = None

def save_model(model, batch_idx, epoch):
  date = time.strftime('%m-%d|%H:%M', time.localtime(time.time()))
  name = f'model_{model_name}_e{epoch}-{batch_idx}_({date}).pt'
  torch.save(model.state_dict(), os.path.join(checkpoint, name))
  return name


def adjust_learningrate(score_list):
  if len(score_list) > 1 and score_list[-1][0] < 0.999 * score_list[-2][0]:
    if restore:
      m_state_dict = torch.load(os.path.join(checkpoint, best_name))
      model.load_state_dict(m_state_dict, strict=False)
    cur_lr_list = []
    for k, group in enumerate(optimizer.param_groups):
      group['lr'] = group['lr'] * 0.5
      cur_lr_list.append(group['lr'])
    cur_lr = ' '.join(map(lambda v: str(v), cur_lr_list))
    print('Current learning rate:', cur_lr)


def sort_batch(batch):
  batch = zip(*batch)
  batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
  batch = zip(*batch)
  return batch


def convert_data(batch, encoder_, device, reverse=False):
  padded = []
  for x in batch:
    encoded = [encoder_.encode(v) for v in x]
    if reverse:
      padded.append(
        [EOS] +
        [item for l in encoded[::-1] for item in l] +
        [SOS])
    else:
      padded.append(
        [SOS] +
        [item for l in encoded for item in l] +
        [EOS])
  max_len = max(len(x) for x in padded)
  # print('max_len:',max_len)
  # print('len before padding:', len(padded[-1]), len(padded[0]))
  padded = [x + [PAD] * max(0, max_len - len(x)) for x in padded]
  # print('len after padding:', len(padded[-1]), len(padded[0]))
  padded = torch.LongTensor(padded).to(device)
  mask = padded.ne(PAD).float()
  return padded, mask


def train(epoch):
  global best_name
  model.train()
  loss = 0
  for batch_idx, batch in enumerate(train_iter, start=1):
    start_time = time.time()
    batch = list(sort_batch(batch))  # sort batch by src length
    src_raw = batch[0]
    trg_raw = batch[1]
    src, src_mask = convert_data(src_raw, encoder_src, device, False)
    f_trg, f_trg_mask = convert_data(trg_raw, encoder_trg, device, False)
    optimizer.zero_grad()
    if cuda and torch.cuda.device_count() > 1 and local_rank is None:
      R = nn.parallel.data_parallel(model, (
        src, src_mask, f_trg, f_trg_mask), device_ids)
    else:
      R = model(src, src_mask, f_trg, f_trg_mask)
    R[0].mean().backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(param_list, grad_clip)
    optimizer.step()
    elapsed = time.time() - start_time
    loss += R[-1].item()
    R = map(lambda x: str(x.mean().item()), R)
    # R: (batch_size, 2), R[:, 0]: total loss, R[:, 1]: average loss of per words
    print(f'epoch: {epoch}, batch: {batch_idx}, elapsed time: {elapsed}s, '
          f'cur_lr: {cur_lr}, loss: ', ' '.join(R))

    # validation
    if batch_idx % vfreq == 0:
      evaluate(batch_idx, epoch)
      model.train()
      if decay_lr:
        adjust_learningrate(score_list)
        # score_list: ((bleu, idx, epoch), (),())
      if len(score_list) == 1 or \
          score_list[-1][0] > max(map(lambda x: x[0], score_list[:-1])):
        if best_name is not None:
          os.remove(os.path.join(checkpoint, best_name))
        best_name = save_model(model, batch_idx, epoch)

  print(f'epoch: {epoch}, avg loss: {loss/batch_idx}')


def evaluate(batch_idx, epoch):
  model.eval()
  start_time = time.time()
  all_to_export = []
  for ix, batch in enumerate(valid_iter, start=1):
    src_raw = list(batch)[0]
    trg_raw = list(batch)[1:]
    src, src_mask = convert_data(src_raw, encoder_src, device, True)

    with torch.no_grad():
      output = model.predict(src, src_mask)
      to_export = dict()
      to_export['ch'] = functools.reduce(lambda u, v: u + ' ' + v, src_raw[0])
      to_export['nbest'] = []
      hyp = encoder_trg.decode(output)
      to_export['nbest'].append((None, hyp.strip()))
      print(to_export)
      all_to_export.append(to_export)
  elapsed = time.time() - start_time
  if not os.path.exists(f'generation/{model_name}'):
    os.mkdir(f'generation/{model_name}')
  path = f'generation/{model_name}/{epoch}-{batch_idx}.nbest.pydict'
  with open(path, 'w') as f:
    f.writelines(f'{item}\n' for item in all_to_export)
  bleu, uncase_bleu = compute_bleu_score(nist_src, nist_ref, path)
  print(
    f'BLEU for {epoch}-{batch_idx}: {bleu}, time: {elapsed}')
  score_list.append((bleu, batch_idx, epoch))


for epoch in range(nepoch):
  train(epoch)
  print('-----------------------------------')
  # evaluate(len(train_iter), epoch)
  # print('-----------------------------------')
  if decay_lr:
    adjust_learningrate(score_list)
  if len(score_list) == 1 or \
      score_list[-1][0] > max(map(lambda x: x[0], score_list[:-1])):
    if best_name is not None:
      os.remove(os.path.join(checkpoint, best_name))
    best_name = save_model(model, len(train_iter), epoch)


best = max(score_list, key=lambda x: x[0])
print('best BLEU {}-{}: {}'.format(best[2], best[1], best[0]))
