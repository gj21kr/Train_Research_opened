import os, glob, gc
import random
import itertools
from collections import abc
from threading import Thread
from queue import Empty, Full, Queue
from multiprocessing.context import SpawnContext
from typing import Callable, Sequence, Optional, Union

import monai
from monai.transforms import apply_transform
from monai.data.meta_obj import MetaObj
from monai.data.utils import set_rnd, worker_init_fn, pickle_operations, dev_collate, first, TraceKeys
from monai.data.thread_buffer import _ProcessThreadContext, buffer_iterator

import torch
from torch import optim
from torch.utils.data.dataloader import default_collate as torch_collate
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import DataLoader as _TorchDataLoader

class Dataset(_TorchDataset):
    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        self.data = data
        if type(transform)==list:
            from monai.transforms import Compose 
            transform = Compose(transform)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        data_i = self.data[index]
        data_i = apply_transform(self.transform, data_i) if self.transform is not None else data_i
        return data_i
        
    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        data = self._transform(index)
        return data 
        
                

def worker_fn(dataset, index_queue, output_queue):
    while True:
        try:
            index = index_queue.get(timeout=0)
        except Empty:
            continue
        if index is None:
            break
        output_queue.put((index, dataset[index]))

def collate_meta_tensor(batch):
    """collate a sequence of meta tensor sequences/dictionaries into
    a single batched metatensor or a dictionary of batched metatensor"""
    if not isinstance(batch, Sequence):
        raise NotImplementedError()
    elem_0 = first(batch)
    
    if isinstance(elem_0, MetaObj):
        collated = torch_collate(batch)
        collated.meta = torch_collate([i.meta or TraceKeys.NONE for i in batch])
        collated.applied_operations = [i.applied_operations or TraceKeys.NONE for i in batch]
        collated.is_batch = True
        return collated
    if isinstance(elem_0, abc.Mapping): # dict
        data = {k: collate_meta_tensor([d[k] for d in batch if k in d.keys()]) for k in elem_0}
        return data
    if isinstance(elem_0, (tuple, list)):
        data = [collate_meta_tensor([d[i] for d in batch if i < len(d)]) for i in range(len(elem_0))]
        return data
    
    data = torch_collate(batch)
    return data


def list_data_collate(batch: Sequence):
    elem = batch[0]
    data = [i for k in batch for i in k if not k is None] if isinstance(elem, list) else batch
    key = None

    try:
        if monai.config.USE_META_DICT:
            data = pickle_operations(data)  # bc 0.9.0
        if isinstance(elem, abc.Mapping):
            ret = {};
            for k in elem:
                key = k
                data_for_batch = [d[key] for d in data]
                ret[key] = collate_meta_tensor(data_for_batch)
        else:
            ret = collate_meta_tensor(data)
        return ret
    except RuntimeError as re:
        re_str = str(re)
        if "equal size" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            
            re_str += (
                # "\n\nMONAI hint: if your transforms intentionally create images of different shapes, creating your "
                # + "`DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem (check its "
                # + "documentation)."
            )
        _ = dev_collate(data)
        raise RuntimeError(re_str) from re
    except TypeError as re:
        re_str = str(re)
        if "numpy" in re_str and "Tensor" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            re_str += (
                "\n\nMONAI hint: if your transforms intentionally create mixtures of torch Tensor and numpy ndarray, "
                + "creating your `DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem "
                + "(check its documentation)."
            )
        _ = dev_collate(data)
        raise TypeError(re_str) from re


class DataLoader(_TorchDataLoader):
    def __init__(self, dataset: _TorchDataset, num_workers: int = 0, **kwargs) -> None:
        if num_workers == 0:
            # when num_workers > 0, random states are determined by worker_init_fn
            # this is to make the behavior consistent when num_workers == 0
            # torch.int64 doesn't work well on some versions of windows
            _g = torch.random.default_generator if kwargs.get("generator") is None else kwargs["generator"]
            init_seed = _g.initial_seed()
            _seed = torch.empty((), dtype=torch.int64).random_(generator=_g).item()
            set_rnd(dataset, int(_seed))
            _g.manual_seed(init_seed)
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = list_data_collate
        if "worker_init_fn" not in kwargs:
            kwargs["worker_init_fn"] = worker_init_fn

        super().__init__(dataset=dataset, num_workers=num_workers, **kwargs)

