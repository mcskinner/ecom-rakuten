from . import bpv, data, scoring
from .slimai import to_np

from kerosene import torch_util

import fire
import numpy as np


def infer(model, dl):
    probs = []
    targs = []
    model.eval()
    for x, y in dl:
        probs.append(to_np(model(torch_util.variable(x))))
        targs.append(to_np(y))
    return np.concatenate(probs), np.concatenate(targs)


def predict(scores, tune_f1=False):
    if not tune_f1:
        return scores.argmax(axis=1)
    probs = scoring.softmax(scores)
    pcuts = scoring.pred_from_probs(probs)
    probs[probs < pcuts] = 0
    probs[:, -1] += 1e-9
    return probs.argmax(axis=1)


def main(forward='', reverse=''):
    n_emb, n_hid = 50, 512
    enc, cenc = data.load_encoders()
    n_inp, n_out = len(enc.itos), len(cenc.itos)

    models_by_dir = {
        False: forward.split(','),
        True: reverse.split(','),
    }

    total_scores, total_targs = None, None
    for is_reverse, models in models_by_dir.items():
        for model_name in models:
            _, val_dl = data.load_dataloaders(is_reverse)
            model = data.load_model(
                torch_util.to_gpu(bpv.BalancedPoolLSTM(n_inp, n_emb, n_hid, n_out)),
                model_name,
            )
            scores, targs = infer(model, val_dl)

            preds = predict(scores)
            print(model_name, is_reverse, scoring.score(preds, targs))

            if total_scores is None:
                total_scores, total_targs = scores, targs
            else:
                assert (targs == total_targs).all()
                total_scores += scoring.log_softmax(scores)

    print(scoring.score(predict(total_scores), total_targs))
    print(scoring.score(predict(scores, tune_f1=True), total_targs))


if __name__ == '__main__':
    fire.Fire(main)
