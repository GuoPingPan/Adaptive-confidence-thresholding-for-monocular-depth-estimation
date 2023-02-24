#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random


def get_transform(method=Image.BICUBIC, normalize=True):
    transform_list = []

    #osize = [192, 640]
    osize = [192, 480]
    transform_list.append(transforms.Resize(osize, method))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


