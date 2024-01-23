import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Dataset import QADataset
from torchvision import transforms
from PIL import ImageFile
from pytorch_lightning.loggers import WandbLogger
import torch
from ViT import *
from ViT_QA import *
from pytorch_lightning.callbacks import ModelCheckpoint
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train():
    # 모델 인스턴스 생성
    #패치 사이즈
    p_s = 16
    model_kwargs = {
        'embed_dim': (p_s*p_s*3),
        'hidden_dim': (p_s*p_s*3)*4,
        'num_channels': 3,
        'num_heads': 8,
        'num_layers': 6,
        'num_classes': 3,
        'patch_size': p_s,
        'num_patches': (128//p_s)**2,
        'dropout': 0.1,
        'head_num_layers': 2 
    }
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
    ## 연구실
    batch_size = 128
    num_workers = 8
    ## 집
    # batch_size = 64
    # num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True, persistent_workers=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers,pin_memory=True, persistent_workers=True)

    #torch.set_float32_matmul_precision('high')
    
    # 체크포인트 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoint/vit",
        filename="ViT_{epoch}-{val_loss:.2f}",
        save_top_k=3,  # 성능이 가장 좋은 상위 3개의 체크포인트만 저장
        monitor="val_loss",  # 모니터링할 메트릭
        mode="min",  # "min"은 val_loss를 최소화하는 체크포인트를 저장
    )

    #model = ViT_trans(model_kwargs, lr=1e-3)
    model = ViT_QA_cos(model_kwargs, lr=1e-3)
    # 트레이너 설정 및 학습
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='auto',
        devices=1,
        log_every_n_steps=20,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
    
if __name__ == '__main__':
    # Windows 환경에서 멀티프로세싱을 사용할 때 필요
    torch.multiprocessing.freeze_support()
    train()

## 연구실 train 시 아래 코드 사용 ##
# python -Xfrozen_modules=off -m train_lbw