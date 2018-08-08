from kerosene.torch_util import to_gpu

import collections
import numpy as np
import os
import torch
import torch.utils.data

from concurrent.futures.thread import ThreadPoolExecutor


class SortSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, key):
        self.data_source, self.key = data_source, key

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))


class SortishSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, key, bs):
        self.data_source, self.key, self.bs = data_source, key, bs

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        sz = self.bs*50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


class DataLoader(object):
    def __init__(
        self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, pad_idx=0,
        num_workers=None, pin_memory=False, drop_last=False, pre_pad=True, half=False,
        transpose=False, transpose_y=False,
    ):
        self.dataset, self.batch_size, self.num_workers = dataset, batch_size, num_workers
        self.pin_memory, self.drop_last, self.pre_pad = pin_memory, drop_last, pre_pad
        self.transpose, self.transpose_y = transpose, transpose_y
        self.pad_idx, self.half = pad_idx, half

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)

        if num_workers is None:
            self.num_workers = num_cpus()

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.batch_sampler)

    def jag_stack(self, b):
        if len(b[0].shape) not in (1, 2):
            return np.stack(b)
        ml = max(len(o) for o in b)
        if min(len(o) for o in b) == ml:
            return np.stack(b)
        res = np.zeros((len(b), ml), dtype=b[0].dtype) + self.pad_idx
        for i, o in enumerate(b):
            if self.pre_pad:
                res[i, -len(o):] = o
            else:
                res[i, :len(o)] = o
        return res

    def np_collate(self, batch):
        b = batch[0]
        if isinstance(b, (np.ndarray, np.generic)):
            return self.jag_stack(batch)
        elif isinstance(b, (int, float)):
            return np.array(batch)
        elif isinstance(b, (str, bytes)):
            return batch
        elif isinstance(b, collections.Mapping):
            return {key: self.np_collate([d[key] for d in batch]) for key in b}
        elif isinstance(b, collections.Sequence):
            return [self.np_collate(samples) for samples in zip(*batch)]
        raise TypeError(("batch must contain numbers, dicts or lists; found {}".format(type(b))))

    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        if self.transpose:
            res[0] = res[0].T
        if self.transpose_y:
            res[1] = res[1].T
        return res

    def __iter__(self):
        if self.num_workers == 0:
            for batch in map(self.get_batch, iter(self.batch_sampler)):
                yield self._get_tensor(batch, self.pin_memory, self.half)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
                for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
                    for batch in e.map(self.get_batch, c):
                        yield self._get_tensor(batch)

    def _get_tensor(self, batch):
        if isinstance(batch, (np.ndarray, np.generic)):
            batch = T(batch, half=self.half, cuda=False).contiguous()
            if self.pin_memory:
                batch = batch.pin_memory()
            return to_gpu(batch)
        elif isinstance(batch, (bytes, str)):
            return batch
        elif isinstance(batch, collections.Mapping):
            return {k: self._get_tensor(sample) for k, sample in batch.items()}
        elif isinstance(batch, collections.Sequence):
            return [self._get_tensor(sample) for sample in batch]
        raise TypeError(f"batch must contain numbers, dicts or lists; found {type(batch)}")


def T(a, half=False, cuda=True):
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else:
            raise NotImplementedError(a.dtype)
    return to_gpu(a, async=True) if cuda else a


def chunk_iter(iterable, chunk_size):
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(iterable))
            yield chunk
        except StopIteration:
            if chunk:
                yield chunk
            break


def to_np(v):
    if isinstance(v, (np.ndarray, np.generic)):
        return v
    if isinstance(v, (list, tuple)):
        return [to_np(o) for o in v]

    if isinstance(v, torch.autograd.Variable):
        v = v.data
    if isinstance(v, torch.cuda.HalfTensor):
        v = v.float()
    return v.cpu().numpy()


def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()
