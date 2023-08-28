import sys
import importlib
from monai.transforms import Compose, ToDeviced

def call_trans_function(config, device=None):
    mod = ''
    try:
        mod = importlib.import_module(f'transforms.trans_v{config["TRANSFORM"]}')
    except:
        mod = importlib.import_module(f'trans_v{config["TRANSFORM"]}')
    train_transforms, val_transforms = mod.call_transforms(config)
    
    if device is not None:
        train_transforms.insert(1, ToDeviced(keys=["image", "label"], device=device))
        val_transforms.insert(1, ToDeviced(keys=["image", "label"], device=device))

    return Compose(train_transforms), Compose(val_transforms)