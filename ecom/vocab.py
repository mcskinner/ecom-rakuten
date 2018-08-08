import numpy as np


class CharEncoder(object):
    def __init__(self, itos, stoi, freq):
        self.itos = itos
        self.stoi = stoi
        self.freq = freq

    def encode(self, items):
        return np.array([
            [1] + [self.stoi.get(ch, 2) for ch in item] + [2]
            for item in items
        ])

    def decode(self, idxs):
        return ''.join([self.itos[i] for i in idxs])


class CategoryEncoder(object):
    def __init__(self, itos, stoi, freq):
        self.itos = itos
        self.stoi = stoi
        self.freq = freq

    def encode(self, cats):
        return np.array([self.stoi[o] for o in cats])

    def decode(self, idxs):
        return [self.itos[i] for i in idxs]


def mk_stoi(itos):
    return {o: i for i, o in enumerate(itos)}
