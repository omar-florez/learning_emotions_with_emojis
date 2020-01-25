import os, sys
sys.path.append(os.getcwd())
from models.BERT import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
import seaborn

import ipdb

class GeneratorClassifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class GPT2(nn.Module):
    def __init__(self, decoder, tgt_embed, generator):
        super().__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, tgt, tgt_mask):
        return self.decode(tgt, tgt_mask)

    def decode(self, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), tgt_mask)


# Unroll the entire decoder stacking the N Transformer modules
class GPT2_Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Build one Transformer module (self-attention + feed-forward layer)
class GPT2_Decoder_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SubLayerConnection implements: x + dropout(sublayer(norm(x)))
        self.sublayers = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # 0: x + norm(self-attention(x))
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 1: x + norm(feed-forward(x))
        return self.sublayers[1](x, lambda x: self.feed_forward(x))


def GPT2_make_model(tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, num_classes=3):
    c = copy.deepcopy
    position = PositionalEncoding(d_model, dropout)
    att = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

    decoder_layer = GPT2_Decoder_Layer(size=d_model,
                                       self_attn=c(att),
                                       feed_forward=c(ff),
                                       dropout=dropout)
    model = GPT2(GPT2_Decoder(decoder_layer, N),
                 tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab),
                                         c(position)),
                 #generator=Generator(d_model, num_classes))
                 generator=GeneratorForClassification(d_model, num_classes))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def make_classification(x, model, num_classes):
    return torch.matmul(model(x), nn.Linear(d_model, num_classes))

def GP2_run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model(batch.trg, batch.trg_mask)
        #loss = loss_compute(out, batch.trg_y, batch.ntokens.float())
        loss = loss_compute(out, batch.labels, batch.ntokens.float())
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch step: {} Loss: {} Tokens per Sec: {}".format(i,
                                                                      loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def sample_decoding(model, max_len, start_symbol, seq_len):
    """ Use the model's decoder to predict the next symbol by sampling from the
        model's  output distribution"""

    # starting rom start_symbol, we'll keep appending to this tensor
    ys = torch.tensor([[start_symbol]])#.cuda()
    for i in range(max_len - 1):
        # make the length of the sequence  what the model was trained for
        y_last = ys[-(seq_len - 2):]

        mask = subsequent_mask(y_last.size(1))#.cuda()
        out = model.decode(y_last, mask)
        prob = model.generator(out[:, -1])
        prob_np = prob.data[0].cpu().numpy()
        prob_np = np.exp(prob_np) / sum(np.exp(prob_np))

        next_word = np.random.choice(len(prob_np), p=prob_np)
        next_word_tns = torch.tensor([[next_word]], dtype=ys.data.dtype)#.cuda()
        ys = torch.cat([ys, next_word_tns], dim=1)
    return ys

def compute_experiments(model, data_iter):
    precision_batches = []
    for i, batch in enumerate(data_iter):
        # out:              torch.Size([32, 9, 512])
        # batch.trg:        torch.Size([32, 9])
        # batch.trg_mask:   torch.Size([32, 9, 9])
        out = model.decode(batch.trg, batch.trg_mask)
        # prob:             torch.Size([32, 3])
        prob = model.generator(out)
        prob_np = prob.data.cpu().numpy()
        prob_np = np.exp(prob_np)/np.sum(np.exp(prob_np))

        pred_index = np.argmax(prob_np, axis=1)
        actual_index = np.argmax(batch.labels.data.cpu().numpy(), axis=1)
        precision = np.equal(pred_index, actual_index).sum()/float(len(pred_index))
        precision_batches.append(precision)
    return np.mean(precision_batches)

#------------------------------------------------------------------------------------------------
# Run
from torchtext import data, datasets
import spacy
from data.EmojiDatasetWords import EmojiDatasetWords

spacy_es = spacy.load('es')
dataset = EmojiDatasetWords()
BLANK_WORD = '<blank>'
MIN_FREQ = 2

def tokenize_es(text):
    return [tok.text for tok in spacy_es.tokenizer(text)]

SRC = data.Field(tokenize=tokenize_es, pad_token=BLANK_WORD)
TGT = data.Field(sequential=False, unk_token=None)

train, val, test = data.TabularDataset.splits(path='data',
                                              train='train.tsv',
                                              validation='val.tsv',
                                              test='test.tsv',
                                              format='TSV',
                                              skip_header=True,
                                              fields=[('text', SRC),
                                                      ('target', TGT)])

SRC.build_vocab(train, val, test, min_freq=MIN_FREQ)
TGT.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test),
                                                             batch_size=32,
                                                             repeat=False,
                                                             shuffle=True,
                                                             sort=False)
                                                             #device=args.gpu,
                                                             #sort_key=sort_key)

print('20 elements of the train vocabulary:')
print(SRC.vocab.itos[:20])
print(list(SRC.vocab.stoi.items())[:20])

seq_len = 32
num_classes = 3
tgt_vocab = len(SRC.vocab)
model = GPT2_make_model(tgt_vocab=tgt_vocab, N=4, d_model=512, d_ff=2048, h=8, dropout=0.2)
#model.to('cuda:0')

# Train model
import warnings
warnings.filterwarnings(action='once')

pad_idx = SRC.vocab.stoi['<blank>']

#criterion = LabelSmoothing(size=len(SRC.vocab), padding_idx=pad_idx, smoothing=1e-6)#.cuda()
#   T = args.temperature if args and hasattr(args, 'temperature') else 4
#   kd_loss = nn.KLDivLoss()(F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1)) * (T * T)
criterion = nn.KLDivLoss(size_average=False)

model_opt = VaryingRateOpt(model_size=model.tgt_embed[0].d_model,
                           factor=1, warmup=1.0,
                           optimizer=torch.optim.Adam(model.parameters(),
                                                      lr=0.0,
                                                      betas=(0.9, 0.98),
                                                      eps=1e-9))



for epoch in range(20):
  model.train()
  GP2_run_epoch((rebatch_classification(pad_idx, b, num_classes) for b in train_iter),
                model,
                SimpleLossCompute(model.generator,      # predicted vector
                                  criterion,            # actual vector
                                  opt=model_opt))

  model.eval()
  loss = GP2_run_epoch((rebatch_classification(pad_idx, b, num_classes) for b in val_iter),
                       model,
                       SimpleLossCompute(model.generator,
                                         criterion,
                                         opt=None))

  val_precision = compute_experiments(model, (rebatch_classification(pad_idx, b, num_classes) for b in val_iter))
  print('Epoch [{}] Validation loss: {} Precision: {}'.format(epoch, loss.item(), val_precision))

torch.save(model.state_dict(), 'model.state')


# !cat requirements.txt | xargs -n 1 pip install
# !pip install torchtext spacy
# !python -m spacy download en
# !python -m spacy download es