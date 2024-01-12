import os
from secrets import choice
import urllib.request
from urllib.error import HTTPError
from xml.dom.pulldom import default_bufsize
import tempfile
from ray import tune
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
# from IPython.display import set_matplotlib_formats
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning.loggers import WandbLogger
from question_loader import *
from torchvision.models import resnet50
from torchmetrics import F1Score
from torchmetrics.functional import f1_score
from models import *
from argparse import ArgumentParser
import os
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


def train_model(model_name,pretrained_filename, args, **kwargs):
    metrics = {"loss": "val_loss", "acc": "val_f1"}
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, f"{model_name}_{args.version}"),
        gpus=[2],
        # auto_select_gpus=True,
        logger = wandb_logger,
        strategy='ddp',
        max_epochs=180,
        fast_dev_run=False,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_f1/dataloader_idx_0"),
            LearningRateMonitor("epoch"),
            # TuneReportCallback(metrics, on="validation_end")
        ],
        )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    print(pretrained_filename, os.path.isfile(pretrained_filename))
    pretrained_filename = pretrained_filename
    if os.path.isfile(pretrained_filename):
        # print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        if model_name == 'ViT':
            model = ViT_CE.load_from_checkpoint(pretrained_filename)
        elif model_name == "ViViT":
            model = ViT_MF_CE.load_from_checkpoint(pretrained_filename)
        elif model_name == "ViViT_SP":
            model = ViT_MF_SP_CE.load_from_checkpoint(pretrained_filename)
        elif model_name == "ResNet_2D":
            model = ResNet_CE.load_from_checkpoint(pretrained_filename)
        elif model_name == "C3D_CE":
            model = C3D_CE.load_from_checkpoint(pretrained_filename)
        elif model_name == "ViT_trans":
            model = ViT_trans.load_from_checkpoint(pretrained_filename)
        # val_result = trainer.test(model, dataloaders=val_loader, verbose=True)
        test_result = trainer.test(model, dataloaders=test_loader, verbose=True)
        
        result = {"test": test_result[0], "val": val_result[0]}
        # all_result = []
        # model.eval()
        # for batch_idx, batch in enumerate(tqdm(test_loader)):
        #     # print(batch_idx)
            
        #     result_dict = model.predict_step(batch, batch_idx)
        #     all_result.append(result_dict)
        # with open(f"{args.version}_result.pickle", "wb") as f:
        #     pickle.dump(all_result, f)
        # preds = []
        # gts = []
        # cate = []
        # for dic in all_result:
        #     preds.extend(dic['preds'].tolist())
        #     gts.extend(dic['gts'].tolist())
        #     cate.extend(dic['categories'].tolist())
        # df_data = [preds, gts, cate]
        # result_df = pd.DataFrame(data=df_data, columns=['preds', 'gts', 'categories'])
        # result_df['IDs'] = test_dataset.dir_names
        # result_df.to_csv(f"{args.version}_result.csv", index=False)
    else:
        
        if model_name == "ViT":
            model = ViT_CE(**kwargs)
        elif model_name == "ViViT":
            model = ViT_MF_CE(**kwargs)
        elif model_name == "ViViT_SP":
            model = ViT_MF_SP_CE(**kwargs)
        elif model_name == "ResNet_2D":
            model = ResNet_CE(**kwargs)
        elif model_name == "C3D_CE":
            model =C3D_CE(**kwargs)
        elif model_name == "ViT_trans":
            model =ViT_trans(**kwargs)
        trainer.fit(model, train_loader, val_dataloaders = [val_loader, test_loader])
        # trainer.fit(model, train_loader, val_dataloaders = val_loader)





if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model_name", type=str, default="ViT", choices=['C3D_CE','ResNet_2D','ViT','ViViT', 'ViViT_SP', 'ViT_trans'])
    parser.add_argument("--version", type=str, default="1")
    parser.add_argument("--pretrained_filename", type=str, default = '')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--split_path", type=str, default='/workspace/vi_e2e/cause_effect_data_split_full.json')
    args = parser.parse_args()
    
    wandb_logger = WandbLogger(project="surro_vis_reasoning",name = f"{args.model_name+'_'+args.version}")

    import yaml
    if args.config_path:
        with open(args.config_path) as f:
            model_kwargs = yaml.safe_load(f)
    else:
        with open(f"./configs/{args.model_name}.yaml") as f:
            model_kwargs = yaml.safe_load(f)

    


    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    # 30 %
    # DATASET_PATH = os.environ.get("PATH_DATASETS", "/data/220804_visreasoning/cause_effect/cause_effect_questions")

    DATASET_PATH = os.environ.get("PATH_DATASETS", "/data/cause_effect/cause_effect_problem")
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")

    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)


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
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    data_split_json_path = args.split_path
    train_dataset = Group2Dataset_E2E(root=DATASET_PATH, index_file_path=data_split_json_path,
                                    mode = 'train', transform=train_transform)
    val_dataset = Group2Dataset_E2E(root=DATASET_PATH, index_file_path=data_split_json_path,
                                    mode = 'valid', transform=test_transform)
    test_dataset = Group2Dataset_E2E(root=DATASET_PATH, index_file_path=data_split_json_path,
                                    mode = 'test', transform=test_transform)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_dataset, batch_size=12, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=12, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=12, shuffle=False, drop_last=False, num_workers=4)

        

    # model, results = train_model(args.model_name,
    #     pretrained_filename = args.pretrained_filename,
    #     model_kwargs = model_kwargs,
    #     lr=args.lr,
    # )

    train_model(args.model_name,
        pretrained_filename = args.pretrained_filename,
        model_kwargs = model_kwargs,
        args = args,
        lr=args.lr,
    )
    
    #TODO implement RayTune (https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html#tune-pytorch-lightning-ref)
    """
    num_samples = 10
    num_epochs = 10
    gpus_per_trial = 0 # set this to higher if using GPU

    
    # search space
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    trainable = tune.with_parameters(
        train_mnist,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial)

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        name="tune_mnist")

    print(analysis.best_config)
    """





    # print("ViT results", results)
