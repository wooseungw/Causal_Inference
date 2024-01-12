import typing

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.distributed as dist
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
from collections import defaultdict

#이미지 패치 임베딩
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x
#어텐션 블록
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
        head_num_layers=2,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
    def get_embedding(self, x):
        # Preprocess input

        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        # print('x.shape',x.shape)
        # print('T', T)
        # print('self.pos_embedding.shape',self.pos_embedding.shape)
        # x = x + self.pos_embedding[:, : T + 1]
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]

        return cls


    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
    
#학습을 위한 코드로 보임
class BaseLighteningClass(pl.LightningModule):

    def _calculate_loss(self, batch, mode="train"):
        imgs_list, labels, categories = batch

        labels -= 1
        preds = self(imgs_list)
        loss = F.cross_entropy(preds, labels)

        f1 = f1_score(preds, labels, num_classes=3)
        
        pred_class = preds.argmax(dim=-1)
        acc = (pred_class == labels).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        self.log("%s_f1" % mode, f1)

        return loss, {"preds": pred_class, "gts": labels, "categories": categories}

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        results = []
        if dataloader_idx == 0:
            _, result = self._calculate_loss(batch, mode="val")
            results.append(result)
        else:
            _, result = self._calculate_loss(batch, mode="test")
            results.append(result)

    
        return results

    def test_step(self, batch, batch_idx):
        _, results = self._calculate_loss(batch, mode="test")
        return results
    
    def predict_step(self, batch, batch_idx):
        imgs_list, labels, categories = batch

        labels -= 1
        preds = self(imgs_list)        
        pred_class = preds.argmax(dim=-1)
        

        return {"preds": pred_class, "gts": labels, "categories": categories}
        

    def validation_epoch_end(self, validation_step_outputs: typing.List[typing.Dict]):
        self._calculate_macro_fscore(validation_step_outputs, mode="val")

    def test_epoch_end(self, validation_step_outputs: typing.List[typing.Dict]):
        self._calculate_macro_fscore_test(validation_step_outputs, mode="test")

    def _calculate_macro_fscore_test(self, step_outputs: typing.List[typing.Dict], mode: str):
        """Uses results from all steps to calculate accuracy per category and macro fscore."""
        result_dict = defaultdict(list)
        # First combine results for all steps based on category
        print('step_outputs:  ', step_outputs)
        for output in step_outputs:
            # print(output)
            # output = output[0]
            # print(output)
            print(output)
            print(type(output))
            for pred, gt, cat in zip(output["preds"].tolist(), output["gts"].tolist(), output["categories"]):
                result_dict[cat].append(pred == gt)
        # NOTE: THESE DISTS CALLS WILL ONLY WORK IF MULTIPLE GPUS ARE IN USE. OTHERWISE THIS CAN RAISE AN ERRO
        dist.barrier()
        outputs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(outputs, dict(result_dict))
        # Ensure only one device calculates the per category accuracy and macro fscore
        if dist.get_rank() == 0:
            all_accuracy = []
            category_results = defaultdict(list)
            # Combine results across gpu devices
            for output in outputs:
                for k, v in output.items():
                    category_results[k] = category_results[k] + v
            # Log accuracies and macro f score.
            for k, v in category_results.items():
                acc = torch.tensor(v).float().mean()
                all_accuracy.append(acc)
                self.log(f'{k}_{mode}_acc', acc, on_step=False, on_epoch=True, rank_zero_only=True)
            self.log(f"{mode}_f1_macro",
                     sum(all_accuracy) / len(all_accuracy),
                     on_step=False,
                     on_epoch=True,
                     rank_zero_only=True)
        return step_outputs

    def _calculate_macro_fscore(self, step_outputs: typing.List[typing.Dict], mode: str):
        """Uses results from all steps to calculate accuracy per category and macro fscore."""
        result_dict = defaultdict(list)
        # First combine results for all steps based on category
        # print('step_outputs:  ',step_outputs)
        for output in step_outputs:
            for sub_output in output:
                for sub_sub_output in sub_output:

                    # print(output)
                    # output = output[0]
                    # print(output)
                    print(sub_sub_output)
                    print(type(sub_sub_output))
                    for pred, gt, cat in zip(sub_sub_output["preds"].tolist(), sub_sub_output["gts"].tolist(), sub_sub_output["categories"]):
                        result_dict[cat].append(pred == gt)
        # NOTE: THESE DISTS CALLS WILL ONLY WORK IF MULTIPLE GPUS ARE IN USE. OTHERWISE THIS CAN RAISE AN ERRO
        dist.barrier()
        outputs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(outputs, dict(result_dict))
        # Ensure only one device calculates the per category accuracy and macro fscore
        if dist.get_rank() == 0:
            all_accuracy = []
            category_results = defaultdict(list)
            # Combine results across gpu devices
            for output in outputs:
                for k, v in output.items():
                    category_results[k] = category_results[k] + v
            # Log accuracies and macro f score.
            for k, v in category_results.items():
                acc = torch.tensor(v).float().mean()
                all_accuracy.append(acc)
                self.log(f'{k}_{mode}_acc', acc, on_step=False, on_epoch=True, rank_zero_only=True)
            self.log(f"{mode}_f1_macro",
                     sum(all_accuracy) / len(all_accuracy),
                     on_step=False,
                     on_epoch=True,
                     rank_zero_only=True)

'''실제 학습에 사용되는 ViT 모델'''
class ViT_trans(BaseLighteningClass):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        ##############################
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_kwargs['embed_dim']))
        self.pos_embedding = nn.Parameter(torch.randn(1, 5, model_kwargs['embed_dim']))
        self.dropout = nn.Dropout(model_kwargs['dropout'])
        self.transformer = nn.Sequential(
            *(AttentionBlock(model_kwargs['embed_dim'], model_kwargs['hidden_dim'], model_kwargs['num_heads'], dropout=model_kwargs['dropout']) for _ in range(model_kwargs['head_num_layers']))
        )
        ##############################
        self.mlp_head = nn.Sequential(nn.LayerNorm(model_kwargs['embed_dim']),
                                      nn.Linear(model_kwargs['embed_dim'], model_kwargs['num_classes']))
        self.f1_cal = F1Score(num_classes=3)

    def forward(self, x):
        x = torch.stack(x).permute(1, 0, 2, 3, 4)
        embeddings_list = []
        for imgs in x:
            embeddings = self.model.get_embedding(imgs)  # shape (4, embedding_size)
            embeddings_list.append(embeddings)
        embeddings = torch.stack(embeddings_list, 0) # bsz, 4, embdding_size(이미지 하나의 vit cls token representation dim)
        ###############################
        B, _, _ = embeddings.shape
        cls_token = self.cls_token.repeat(B, 1, 1) # [B, 1, embed_dim]
        embeddings = torch.cat([cls_token, embeddings], dim=1) # [B, 5, embed_dim]
        embeddings = embeddings + self.pos_embedding
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.transpose(0, 1) # [5, B, embed_dim]
        embeddings = self.transformer(embeddings)
        cls = embeddings[0]
        ################################
        preds = self.mlp_head(cls) # bsz, num_classes
        return preds

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)
        return [optimizer], [lr_scheduler]