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

from torchprofile import profile_macs
#


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

class ViViT_SP(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        # self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        # cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_temporal_tokens, x), dim=1)

        # x = self.temporal_transformer(x)
        # 

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
    



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


class ViT(BaseLighteningClass):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.mlp_head = nn.Sequential(nn.LayerNorm(model_kwargs['embed_dim']),
                                      nn.Linear(model_kwargs['embed_dim'], model_kwargs['num_classes']))
        # TODO: This train_loader is unreferenced amd will cause an erro
        self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]


class ViT_CE(BaseLighteningClass):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        # self.example_input_array = next(iter(train_loader))[0]
        self.mlp_head = nn.Sequential(nn.LayerNorm(model_kwargs['embed_dim']*4),
                                      nn.Linear(model_kwargs['embed_dim']*4, model_kwargs['num_classes']))
        self.f1_cal = F1Score(num_classes=3)

    def forward(self, x):
        x = torch.stack(x).permute(1, 0, 2, 3, 4)
        embeddings_list = []
        for imgs in x:
            # imgs = torch.stack(imgs)
            # print(imgs.shape)
            embeddings = self.model.get_embedding(imgs)  # shape (4, embedding_size)
            # print(embeddings.shape)
            embeddings_list.append(embeddings.reshape(4*embeddings.shape[1]))
        embeddings = torch.stack(embeddings_list) # bsz, embdding_size(이미지 하나의 vit cls token representation dim * 4)
        preds = self.mlp_head(embeddings) # bsz, num_classes
        return preds

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)
        return [optimizer], [lr_scheduler]


class ViT_MF_CE(BaseLighteningClass):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        # ViViT
        self.save_hyperparameters()
        self.model = ViViT(**model_kwargs) 

    def forward(self, x):
        x = torch.stack(x).permute(1, 0, 2, 3, 4) 
        preds = self.model(x)
        
        return preds

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)
        return [optimizer], [lr_scheduler]


class ViT_MF_SP_CE(BaseLighteningClass):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        # ViViT
        self.save_hyperparameters()
        self.model = ViViT_SP(**model_kwargs) 

    def forward(self, x):
        x = torch.stack(x).permute(1, 0, 2, 3, 4) 
        preds = self.model(x)
        
        return preds

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)
        return [optimizer], [lr_scheduler]


class ResNet_CE(BaseLighteningClass):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet50()
        self.model.conv1 = nn.Conv2d(3*model_kwargs["num_frames"],  64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features = model_kwargs["num_classes"])

    def forward(self, x):
        x = torch.stack(x).permute(1, 0, 2, 3, 4)
        b, t, c, h, w = x.shape

        x = x.reshape(b, t*c, h, w)

        preds = self.model(x)

        return preds

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)
        return [optimizer], [lr_scheduler]


class C3D_CE(BaseLighteningClass):
    '''
    conv1  in:16*3*112*112   out:16*64*112*112
    pool1  in:16*64*56*56    out:16*64*56*56
    conv2  in:16*64*56*56    out:16*128*56*56
    pool2  in:16*128*56*56   out:8*128*28*28
    conv3a in:8*128*28*28    out:8*256*28*28
    conv3b in:8*256*28*28    out:8*256*28*28
    pool3  in:8*256*28*28    out:4*256*14*14
    conv4a in:4*512*14*14    out:8*512*14*14
    conv4b in:4*512*14*14    out:8*512*14*14
    pool4  in:4*512*14*14    out:2*512*7*7
    conv5a in:2*512*7*7      out:2*512*7*7
    conv5b in:2*512*7*7      out:2*512*7*7
    pool5  in:2*512*7*7      out:1*512*4*4
    '''

    def __init__(self,model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        # self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(32768, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 3)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def init_weight(self):
        for name, para in self.named_parameters():
            if name.find('weight') != -1:
                nn.init.xavier_normal_(para.data)
            else:
                nn.init.constant_(para.data, 0)

    def forward(self, x):
        x = torch.stack(x).permute(1, 2, 0, 3, 4)
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        # h = self.relu(self.conv5a(h))
        # h = self.relu(self.conv5b(h))
        # h = self.pool5(h)
        # print(h.shape)

        h = h.view(-1, 32768)

        h = self.relu(self.fc6(h))
        h = self.dropout(h)

        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)

        return logits

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)
        return [optimizer], [lr_scheduler]


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
        self.f1_cal = F1Score(num_classes=3,task = 'multiclass')

    def forward(self, x):
        if isinstance(x, list) and all(isinstance(item, torch.Tensor) for item in x):
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


if __name__ == "__main__":
    
    img = torch.ones([1, 4, 3, 128, 128])
    model_kwargs={
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'patch_size': 4,
        'num_channels': 3,
        'num_patches': 4097,
        'num_classes': 3,
        'dropout': 0.1,
        'head_num_layers': 2
    }
    model = ViT_trans(model_kwargs,lr=1e-3)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    flops = profile_macs(model, img)
    print("flops:",flops)
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    
    import time
    # 모델을 평가 모드로 설정
    model.eval()

    # 추론 시간 측정
    start_time = time.time()
    with torch.no_grad():
        output = model(img)
    end_time = time.time()

    # 추론에 걸린 시간 계산
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time} seconds")