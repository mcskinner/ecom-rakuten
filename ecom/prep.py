#!/usr/bin/env python

from . import data, vocab

import collections
import pandas as pd

from sklearn.model_selection import train_test_split


def char_encoder(items):
    freq = collections.Counter(ch for item in items for ch in item)
    itos = [o for o, c in freq.most_common() if c >= 10]
    itos.insert(0, '_PAD_')
    itos.insert(1, '_BOS_')
    itos.insert(2, '_EOS_')
    itos.insert(3, '_UNK_')
    stoi = collections.defaultdict(lambda: 3, vocab.mk_stoi(itos))
    return vocab.CharEncoder(itos, stoi, freq)


def category_encoder(cats):
    freq = collections.Counter(cats)
    itos = [o for o, c in freq.most_common()]
    return vocab.CategoryEncoder(itos, vocab.mk_stoi(itos), freq)


def validation_set(all_train):
    freq = all_train.groupby('cat', as_index=False).count()
    singletons = freq[freq.item == 1].cat
    single_train = all_train[all_train.cat.isin(singletons)]
    multi_train = all_train[~all_train.cat.isin(singletons)]

    most_train, val = train_test_split(
        multi_train,
        test_size=200000,
        random_state=42,
        stratify=multi_train['cat'],
    )
    train = pd.concat([most_train, single_train])
    return train, val


def main():
    train, val = validation_set(data.load_all_train())
    data.save_train_val(train, val)

    enc = char_encoder(train.item)
    cenc = category_encoder(train.cat)
    data.save_encoders(enc, cenc)

    trn_enc = enc.encode(train.item)
    val_enc = enc.encode(val.item)

    trn_ds = data.RakDataset(trn_enc, cenc.encode(train.cat))
    val_ds = data.RakDataset(val_enc, cenc.encode(val.cat))
    data.save_datasets(trn_ds, val_ds)


if __name__ == '__main__':
    main()
