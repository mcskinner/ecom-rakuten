import numpy as np
import warnings

from sklearn.metrics import precision_recall_fscore_support as fscore


def score(preds, targs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, r, f1, _ = fscore(targs, preds, pos_label=None, average='weighted')
    return p, r, f1


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
