import os, glob, sys, copy
import random
sys.path.append('../')
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from copy import copy, deepcopy
from joblib import Parallel, delayed

from torch.utils.data import DataLoader as _TorchDataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate as torch_collate
from torch.utils.data.distributed import DistributedSampler

from multiprocessing.context import SpawnContext
from queue import Empty, Full, Queue
from threading import Thread

from monai.utils import first
from monai.data import SmartCacheDataset, CacheDataset, IterableDataset
from monai.data import DataLoader as _MonaiDataLoader
from monai.data import ThreadDataLoader as _MonaiThreadDataLoader
from monai.data.utils import worker_init_fn

from dataset_loader import DataLoader as _CustomDataLoader
from dataset_loader import ThreadDataLoader as _CustomThreadDataLoader
from dataset_loader import Dataset as _CustomDataset
    
    
def call_fold_dataset(list_raw, target_fold, total_folds=5):
    train, valid = [],[]
    list_ = list_raw.copy()

    folds = list(range(0,total_folds))
    folds = folds+folds[::-1]

    data_split = dict.fromkeys(range(0,total_folds),[])
    for fold in folds*1000:
        data_split[fold] = data_split[fold] + [list_.pop(0) ]
        if len(list_)==0: break

    valid = data_split[target_fold]
    for fold in range(0,total_folds):
        if target_fold == fold : continue
        train += data_split[fold]
    return train, valid

def call_dataloader(config, data_list, transforms, mode):
    if mode == "Train":
        batch_size = config["BATCH_SIZE"]
        ratio=config["DATA_SPLIT"]
        drop_last = True
        shuffle = True
    else :
        batch_size = 1
        ratio = 1
        drop_last = False
        shuffle = False

    ds = Dataset( #SubsamplingDataset(
        data=data_list, transform=transforms, #num_workers=10
    )   

    loader = _TorchDataLoader(
        ds, batch_size=batch_size, num_workers=config["WORKERS"], shuffle=shuffle, pin_memory=False
    )
    return loader

