from . import vocab
from .slimai import DataLoader, SortSampler, SortishSampler

import collections
import csv
import numpy as np
import pandas as pd
import pathlib
import pickle
import torch
import torch.utils.data

DATA_PATH = pathlib.Path('data')
MODEL_PATH = pathlib.Path('data/models')


class RakDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, idx):
        return [np.array(self.x[idx]), np.array(self.y[idx])]

    def __len__(self):
        return len(self.x)


def load_all_train():
    return pd.read_csv(
        DATA_PATH/'rdc-catalog-train.tsv',
        sep='\t',
        header=None,
        names=('item', 'cat'),
    )


def save_train_val(train, val):
    train.to_csv(DATA_PATH/'train.csv', index=False)
    val.to_csv(DATA_PATH/'val.csv', index=False)


def load_train_val():
    train = pd.read_csv(DATA_PATH/'train.csv')
    val = pd.read_csv(DATA_PATH/'val.csv')
    return train, val


def load_encoders():
    with open(DATA_PATH/'encodings.pkl', 'rb') as pf:
        lh, rh = pickle.load(pf)
    ch_itos, ch_freq = lh
    cat_itos, cat_freq = rh

    return (
        vocab.CharEncoder(ch_itos, vocab.mk_stoi(ch_itos), ch_freq),
        vocab.CategoryEncoder(cat_itos, vocab.mk_stoi(cat_itos), cat_freq),
    )


def save_encoders(enc, cenc):
    lh = (enc.itos, enc.freq)
    rh = (cenc.itos, cenc.freq)
    with open(DATA_PATH/'encodings.pkl', 'wb') as pf:
        pickle.dump((lh, rh), pf)


def save_datasets(trn_ds, val_ds):
    lh = (trn_ds.x, trn_ds.y)
    rh = (val_ds.x, val_ds.y)
    with open(DATA_PATH/'datasets.pkl', 'wb') as pf:
        pickle.dump((lh, rh), pf)


def _reverse_all(xs):
    return np.array([list(reversed(x)) for x in xs])


def load_datasets(reverse=False):
    with open(DATA_PATH/'datasets.pkl', 'rb') as pf:
        lh, rh = pickle.load(pf)
    lx, ly = lh
    rx, ry = rh
    if reverse:
        lx, rx = _reverse_all(lx), _reverse_all(rx)
    return RakDataset(lx, ly), RakDataset(rx, ry)


def load_dataloaders(reverse=False, bs=256):
    enc, cenc = load_encoders()
    trn_ds, val_ds = load_datasets(reverse=reverse)
    trn_enc, val_enc = trn_ds.x, val_ds.x

    trn_samp = SortishSampler(trn_enc, key=lambda x: len(trn_enc[x]), bs=bs)
    val_samp = SortSampler(val_enc, key=lambda x: len(val_enc[x]))

    trn_dl = DataLoader(trn_ds, bs, transpose=True, pad_idx=0, pre_pad=False, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, pad_idx=0, pre_pad=False, sampler=val_samp)
    return trn_dl, val_dl


def load_test_ds(reverse=False):
    enc, _ = load_encoders()
    test = pd.read_csv(DATA_PATH/'rdc-catalog-test.tsv', sep='\t', header=None, names=('item',))
    test_enc = enc.encode(test.item)
    if reverse:
        test_enc = _reverse_all(test_enc)
    return RakDataset(test_enc, np.zeros(test_enc.shape[0]))


def load_test_dataloader(reverse=False, bs=256):
    test_ds = load_test_ds(reverse=reverse)
    test_enc = test_ds.x
    test_idx = sorted(range(len(test_enc)), key=lambda i: len(test_enc[i]), reverse=True)
    index = {idx: i for i, idx in enumerate(test_idx)}
    test_revidx = [index[i] for i in range(len(test_enc))]
    test_dl = DataLoader(test_ds, bs, transpose=True, pad_idx=0, pre_pad=False, shuffle=False)
    return test_dl, test_revidx


def save_test_pred(cenc, pred, tune_f1=False):
    test_cats = cenc.decode(pred)
    with open(DATA_PATH/'rdc-catalog-test.tsv') as tf:
        test_items = [l.strip('\n') for l in tf.readlines()]
    test_df = pd.DataFrame(collections.OrderedDict(item=test_items, cat=test_cats))
    path = DATA_PATH/'test-pred{}.tsv'.format('' if tune_f1 else '-notune')
    test_df.to_csv(
        path,
        sep='\t',
        header=None,
        quoting=csv.QUOTE_NONE,
        index=False,
    )
    return path


def save_model(model, name):
    torch.save(model.state_dict(), MODEL_PATH/f'{name}.h5')


def load_model(model, name):
    state = torch.load(MODEL_PATH/f'{name}.h5', map_location=lambda s, _: s)
    model.load_state_dict(state)
    return model
