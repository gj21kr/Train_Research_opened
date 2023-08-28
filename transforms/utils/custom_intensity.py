"""
A collection of "vanilla" transforms for intensity adjustment
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Hashable, Mapping, Sequence
from functools import partial
from typing import Any
from warnings import warn

import numpy as np
import torch

from monai.transforms.transform import RandomizableTransform, MapTransform, Transform
from monai.utils.enums import TransformBackends
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.utils import is_positive
from monai.utils import convert_data_type, convert_to_tensor, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix

DEFAULT_POST_FIX = PostFix.meta()

"""
	Add your own code here!
"""
