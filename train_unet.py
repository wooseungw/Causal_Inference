import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Dataset import QADataset
from torchvision import transforms
from PIL import ImageFile
from pytorch_lightning.loggers import WandbLogger
import torch
from unet import Unet_pl

from pytorch_lightning.callbacks import Callback


ImageFile.LOAD_TRUNCATED_IMAGES = True
def train():
    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='casual_inference')
    
    test_transform = transforms.Compose(
        [
            transforms.Resize((128,128)),
            transforms.ToTensor(),
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
    train_dataset = QADataset(transform = train_transform, loc = dir)
    print(len(train_dataset))
    val_dataset = QADataset(transform = test_transform, loc = dir, istrain =  False)
    print(len(val_dataset))

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True,num_workers=6,pin_memory=True, persistent_workers=True) 
    val_loader = DataLoader(val_dataset, batch_size=32,num_workers=6,pin_memory=True, persistent_workers=True)

    model_kwargs = {
    'num_classes': 3,
    'dropout': 0.1,
    }
    model = Unet_pl(model_kwargs, lr=1e-3)

    # 트레이너 설정 및 학습
    trainer = pl.Trainer(
    max_epochs=20,
    accelerator='auto',
    devices=1,
    log_every_n_steps=10,
    logger=wandb_logger
)
    trainer.fit(model, train_loader, val_loader)
    
if __name__ == '__main__':
    # Windows 환경에서 멀티프로세싱을 사용할 때 필요
    torch.multiprocessing.freeze_support()
    train()