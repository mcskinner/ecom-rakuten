from . import prep
from .slimai import DataLoader


def main():
    bs = 256
    test_ds = prep.load_test_ds()
    test_enc = test_ds.x
    test_idx = sorted(range(len(test_enc)), key=lambda i: len(test_enc[i]), reverse=True)
    blah = {idx: i for i, idx in enumerate(test_idx)}
    test_revidx = [blah[i] for i in range(len(test_enc))]
    test_dl = DataLoader(test_ds, bs, transpose=True, pad_idx=0, pre_pad=False, shuffle=False)
    print(test_dl, test_revidx)
