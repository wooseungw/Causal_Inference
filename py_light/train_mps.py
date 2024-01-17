import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from py_light.Dataset import QADataset
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from py_light.train import train

if __name__ == '__main__':
    train()