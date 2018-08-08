from .slimai import to_np

import numpy as np
import torch
import warnings

from sklearn.metrics import precision_recall_fscore_support as fscore


def accuracy(preds, targs):
    preds = torch.max(preds, dim=1)[1]
    return (preds == targs).float().mean()


def score(gold, pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, r, f1, _ = fscore(gold, pred, pos_label=None, average='weighted')
    accuracy = np.mean(gold == pred)
    return accuracy, p, r, f1


def score_learner(cenc, learn):
    pred, gold = learn.predict_with_targs()
    pred_cats = cenc.decode(pred.argmax(axis=1))
    gold_cats = cenc.decode(gold)
    return score(gold_cats, pred_cats)


def logprob_scale(x):
    return x - np.max(x, axis=1, keepdims=True)


def softmax(x):
    e_x = np.exp(logprob_scale(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def log_softmax(x):
    return np.log(softmax(x))


def run_ensemble(*ensemble, smax=False, with_p=False, is_test=True):
    total_pred = None
    for learn in ensemble:
        pred = learn.predict(is_test=is_test)
        if smax:
            pred = softmax(pred)
        total_pred = pred if total_pred is None else total_pred+pred
    return total_pred


def pred_from_probs(probs):
    prob_freq = np.sum(probs, axis=0)
    n_out = prob_freq.shape[0]
    pcuts = np.zeros(n_out)
    for i in range(n_out):
        ps = probs[:, i]
        ps = ps[ps.argsort()][::-1]
        cp = np.cumsum(ps)
        mass = prob_freq[i]
        prec = cp / (1+np.arange(cp.shape[0]))
        rec = cp / mass
        f1 = 2*prec*rec / (prec+rec)
        pcuts[i] = ps[np.argmax(f1)]
    return pcuts


def cut_score(the_score, val_dl):
    gold = to_np(torch.cat([y for x, y in val_dl]))
    probs = the_score / the_score.sum(axis=1, keepdims=True)
    pcuts = pred_from_probs(probs)
    probs[probs < pcuts] = 0
    probs[:, -1] += 1e-9
    return score(gold, np.argmax(probs, axis=1))
