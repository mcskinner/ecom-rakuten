#!/usr/bin/env python

from . import bpv, data
from .scoring import accuracy

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
    enc, cenc = data.load_encoders()
    trn_dl, val_dl = data.load_dataloaders(reverse=reverse)

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
