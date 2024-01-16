import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Dataset import QADataset
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize([0.7760, 0.7491, 0.7213], [0.2949, 0.3032, 0.3314]),
    ]
)
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.7760, 0.7491, 0.7213], [0.2949, 0.3032, 0.3314]),
    ]
)

dir = os.path.join(os.getcwd(),"Dataset/Train/Image")
train_dataset = QADataset(transform=train_transform, loc = dir)
print(len(train_dataset))
val_dataset = QADataset(transform = test_transform, loc = dir, istrain =  False)
print(len(val_dataset))

from torch.utils.data import DataLoader
# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True) 
val_loader = DataLoader(train_dataset, batch_size=64)

import torch
from ViT import ViT_trans

#torch.set_float32_matmul_precision('high')
# 모델 인스턴스 생성
model_kwargs = {
    'embed_dim': 128,
    'hidden_dim': 512,
    'num_channels': 3,
    'num_heads': 8,
    'num_layers': 6,
    'num_classes': 3,
    'patch_size': 16,
    'num_patches': 64,
    'dropout': 0.1,
    'head_num_layers': 2 
}

model = ViT_trans(model_kwargs, lr=1e-3)

# 트레이너 설정 및 학습
   
trainer = pl.Trainer(max_epochs=10, accelerator='auto', devices=1)

trainer.fit(model, train_loader, val_loader)
