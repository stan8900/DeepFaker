import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.utils as vutils

image_size = 64
batch_size = 128
nz = 100  # Размер входного вектора шума
ngf = 64  # Количество фильтров в генераторе
ndf = 64  # Количество фильтров в дискриминаторе
nc = 3    # Количество цветовых каналов (RGB)

