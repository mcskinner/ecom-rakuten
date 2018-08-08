#!/usr/bin/env python

from . import data
from .scoring import accuracy
from .slimai import DataLoader, SortSampler, SortishSampler

from kerosene import batches, loop, optimizer, sched, torch_util

import fire
import torch
import torch.utils.data
import torch.nn.functional as F


class BalancedPoolLSTM(torch.nn.Module):
    def __init__(self, n_inp, em_sz, nh, out_cat, nl=2, **kwargs):
        super().__init__()
        emb_do = kwargs.pop('emb_do', 0.15)
        rnn_do = kwargs.pop('rnn_do', 0.25)
        out_do = kwargs.pop('out_do', 0.35)
        do_scale = kwargs.pop('do_scale', 1)
        self.nl, self.nh, self.out_cat = nl, nh, out_cat
        self.emb = torch.nn.Embedding(n_inp, em_sz, padding_idx=0)
        self.emb_drop = torch.nn.Dropout(emb_do*do_scale)
        self.rnn = torch.nn.LSTM(em_sz, nh, num_layers=nl, dropout=rnn_do*do_scale)
        self.out_drop = torch.nn.Dropout(out_do*do_scale)
        self.out = torch.nn.Linear(4*nh, out_cat)

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, inp):
        sl, bs = inp.size()
        emb = self.emb_drop(self.emb(inp))
        rnn_out, h = self.rnn(emb, None)
        avgpool = self.pool(rnn_out, bs, False)
        maxpool = self.pool(rnn_out, bs, True)
        minpool = -self.pool(-rnn_out, bs, True)
        x = torch.cat([rnn_out[-1], avgpool, maxpool, minpool], 1)
        return self.out(self.out_drop(x))


def train(
    model_name,
    reverse=False,
    n_epochs=40,
    lr=0.8,
    lr_factor=20,
    mom_hi=0.95,
    mom_lo=0.85,
    n_emb=50,
    n_hid=512,
):
    # Data loading
    enc, cenc = data.load_encoders()
    trn_ds, val_ds = data.load_datasets(reverse=reverse)
    trn_enc, val_enc = trn_ds.x, val_ds.x

    bs = 256
    trn_samp = SortishSampler(trn_enc, key=lambda x: len(trn_enc[x]), bs=bs)
    val_samp = SortSampler(val_enc, key=lambda x: len(val_enc[x]))

    trn_dl = DataLoader(trn_ds, bs, transpose=True, pad_idx=0, pre_pad=False, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, pad_idx=0, pre_pad=False, sampler=val_samp)

    # Training
    n_inp = len(enc.itos)
    n_out = len(cenc.itos)
    model = torch_util.to_gpu(BalancedPoolLSTM(n_inp, n_emb, n_hid, n_out))

    n_batches = n_epochs * len(trn_dl)
    optim = optimizer.make(torch.optim.SGD, model, lr)
    schedule = sched.one_cycle(optim, n_batches, lr_factor=lr_factor, momentums=(mom_hi, mom_lo))
    mgr = batches.Manager(
        model,
        optim,
        F.cross_entropy,
        schedule=schedule,
        metrics=[accuracy],
        seq_first=True,
    )
    print(loop.fit(mgr, trn_dl, val_dl, n_epochs))
    data.save_model(model, model_name)


if __name__ == '__main__':
    fire.Fire(train)
