import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Dataset import QADataset
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from train import train

if __name__ == '__main__':
    train()