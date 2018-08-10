#!/usr/bin/env python

from . import bpv, data
from .scoring import accuracy
from .slimai import DataLoader, SortSampler, SortishSampler

from kerosene import batches, loop, optimizer, sched, torch_util

import fire
import torch
import torch.utils.data
import torch.nn.functional as F


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
    model = torch_util.to_gpu(bpv.BalancedPoolLSTM(n_inp, n_emb, n_hid, n_out))

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
