#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dongyu Zhang
"""
import torch
from torch import nn
from torch.nn import Parameter, BCEWithLogitsLoss
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from torch.nn import LayerNorm as BertLayerNorm
import numpy as np

class PatientLevelEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        mode = config.embed_mode
        assert mode in ["all","note","chunk","no"]
        if mode in ("all","note"):
            self.note_embedding = nn.Embedding(config.max_note_position_embedding, config.hidden_size)
        if mode in ("all","chunk"):
            self.chunk_embedding = nn.Embedding(config.max_chunk_position_embedding, config.hidden_size)
        # combine dims: inputs + optional embeddings
        tot = config.hidden_size
        if mode=="all": tot *= 3
        elif mode in ("note","chunk"): tot *=2
        self.combine = nn.Linear(tot, config.hidden_size) if mode!="no" else None
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs, note_ids=None, chunk_ids=None):
        mode = self.config.embed_mode
        if mode == "all":
            ne = self.note_embedding(note_ids)
            ce = self.chunk_embedding(chunk_ids)
            x = torch.cat((inputs, ne, ce), dim=2)
            out = self.combine(x)
        elif mode == "note":
            ne = self.note_embedding(note_ids)
            out = self.combine(torch.cat((inputs, ne), dim=2))
        elif mode == "chunk":
            ce = self.chunk_embedding(chunk_ids)
            out = self.combine(torch.cat((inputs, ce), dim=2))
        else:
            out = inputs
        if mode!="no":
            out = self.LayerNorm(out)
            out = self.dropout(out)
        return out

class LSTMLayer(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.config = config

        # same embedding logic as your FTLSTMLayer
        self.embeddings = PatientLevelEmbedding(config)

        # bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fct  = BCEWithLogitsLoss()

        # initialize weights like BERT does
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights to match BERT’s default scheme """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, embeds, note_ids=None, chunk_ids=None, labels=None):
        """
        embeds:       FloatTensor [batch, seq_len, hidden_size] from BERT’s pooled output
        note_ids:     LongTensor [batch, seq_len]  (optional positional embeddings)
        chunk_ids:    LongTensor [batch, seq_len]
        labels:       FloatTensor [batch]         (0/1 targets)
        """
        # 1) add your note+chunk embeddings
        x = self.embeddings(embeds, note_ids, chunk_ids)  # [B, T, H]

        # 2) run through LSTM
        batch_size = x.size(0)
        # init hidden and cell: (num_layers*2, batch, hidden_size/2)
        h0 = x.new_zeros(self.config.lstm_layers * 2,
                         batch_size,
                         self.config.hidden_size // 2)
        c0 = x.new_zeros(self.config.lstm_layers * 2,
                         batch_size,
                         self.config.hidden_size // 2)

        lstm_out, _ = self.lstm(x, (h0, c0))              # [B, T, H]
        
        # 3) grab the *last* timestep’s hidden (already bidirectional concat → H)
        final_h = lstm_out[:, -1, :]                     # [B, H]
        final_h = self.dropout(final_h)

        # 4) project to a single logit and apply loss if needed
        logits = self.classifier(final_h).squeeze(-1)     # [B]
        
        if labels is not None:
            loss  = self.loss_fct(logits, labels.float())
            probs = torch.sigmoid(logits)
            return loss, probs
        
        return torch.sigmoid(logits)

class FTLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, config):
        super().__init__()
        self.hidden_size = hidden_size
        # weights for gates
        self.Wi = Parameter(torch.empty(input_size, hidden_size))
        self.Ui = Parameter(torch.empty(hidden_size, hidden_size))
        self.bi = Parameter(torch.zeros(hidden_size))
        self.Wf = Parameter(torch.empty(input_size, hidden_size))
        self.Uf = Parameter(torch.empty(hidden_size, hidden_size))
        self.bf = Parameter(torch.zeros(hidden_size))
        self.Wo = Parameter(torch.empty(input_size, hidden_size))
        self.Uo = Parameter(torch.empty(hidden_size, hidden_size))
        self.bo = Parameter(torch.zeros(hidden_size))
        self.Wc = Parameter(torch.empty(input_size, hidden_size))
        self.Uc = Parameter(torch.empty(hidden_size, hidden_size))
        self.bc = Parameter(torch.zeros(hidden_size))
        # time decay params
        self.a = Parameter(torch.tensor(1.0))
        self.b = Parameter(torch.tensor(1.0))
        self.c = Parameter(torch.tensor(0.02))
        self.k = Parameter(torch.tensor(2.9))
        self.d = Parameter(torch.tensor(4.5))
        self.n = Parameter(torch.tensor(2.5))
        self.ones = None
        self._init_weights(config)

    def _init_weights(self, config):
        for w in [self.Wi,self.Ui,self.Wf,self.Uf,self.Wo,self.Uo,self.Wc,self.Uc]:
            nn.init.normal_(w, mean=0.0, std=config.initializer_range)

    def decay(self, t):
        # flexible time decay
        t = t.unsqueeze(-1)
        T1 = 1/(self.a * (t**self.b) + 1e-8)
        T2 = self.k - self.c * t
        T3 = 1/(1 + (t/self.d)**self.n)
        T = (T1 + T2 + T3)/3
        return T

    def forward(self, x, hx, t):
        h_prev, c_prev = hx
        # apply decay on cell
        if self.ones is None:
            self.ones = torch.ones_like(c_prev)
        decay = self.decay(t)
        c_st = torch.tanh(c_prev)
        c_prev = c_prev - c_st + decay * c_st
        # gates
        i = torch.sigmoid(x @ self.Wi + h_prev @ self.Ui + self.bi)
        f = torch.sigmoid(x @ self.Wf + h_prev @ self.Uf + self.bf)
        o = torch.sigmoid(x @ self.Wo + h_prev @ self.Uo + self.bo)
        c = torch.tanh(x @ self.Wc + h_prev @ self.Uc + self.bc)
        c_new = f*c_prev + i*c
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class FTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, config, bidirectional=True):
        super().__init__()
        self.cell = FTLSTMCell(input_size, hidden_size, config)
        self.bidirectional = bidirectional

    def forward(self, seq, times):
        # seq: [batch, chunks, hidden]
        batch,ch,_ = seq.size()
        h = torch.zeros(batch, self.cell.hidden_size, device=seq.device)
        c = torch.zeros(batch, self.cell.hidden_size, device=seq.device)
        outs = []
        for step in range(ch):
            h,c = self.cell(seq[:,step,:], (h,c), times[:,step])
            outs.append(h)
        out = torch.stack(outs, dim=1)
        if self.bidirectional:
            # backward pass
            hb = torch.zeros_like(h); cb = torch.zeros_like(c)
            outsb = []
            for step in reversed(range(ch)):
                hb,cb = self.cell(seq[:,step,:], (hb,cb), times[:,step])
                outsb.append(hb)
            outsb = list(reversed(outsb))
            out = torch.cat((out, torch.stack(outsb,dim=1)), dim=2)
            h = torch.cat((h,hb), dim=1)
        return out, (h,c)

class FTLSTMLayer(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.config = config
        D = config.hidden_size
        self.emb = PatientLevelEmbedding(config)
        self.ftlstm = FTLSTM(D, D//2, config, bidirectional=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(D, num_labels)
        self.loss_fct = BCEWithLogitsLoss()

    def forward(self, embeds, times, note_ids=None, chunk_ids=None, labels=None):
        x = self.emb(embeds, note_ids, chunk_ids)
        out, (h, _) = self.ftlstm(x, times)
        # take last timestep representation
        final = out[:,-1,:]
        final = self.dropout(final)
        logits = self.classifier(final).squeeze(-1)
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())
            probs = torch.sigmoid(logits)
            return loss, probs
        else:
            return torch.sigmoid(logits)
