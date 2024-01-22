import typing

from BaseLightningClass import BaseLightningClass
from module import Attention, PreNorm, FeedForward

import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from torch import nn, einsum
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import f1_score
from torchmetrics import F1Score
import torch.distributed as dist
from collections import defaultdict
from torchvision import transforms, datasets
import timm  # PyTorch Image Models 라이브러리
import numpy as np


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
        #[0]번은 어텐션 벨류, [1]번은 어텐션 가중치
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
    
class CrossAttentionBlock(nn.Module):
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
        y = x[1]
        x = x[0]
        inp_x = self.layer_norm_1(x)
        inp_y = self.layer_norm_1(y)
        x = x + self.attn(inp_y, inp_x, inp_x)[0]
        #print("Cross attention",x.shape)
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

        self.embedding = nn.Sequential(
            nn.Conv2d(3,embed_dim,kernel_size=patch_size,stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        # Layers/Networks
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.ffn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
    def get_embedding(self, x):
        # Preprocess input
        x = self.embedding(x)
        B, T, _ = x.shape

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
        #print("Vit의 cls 모양",cls.shape)
        return cls

    def get_value(self,x):
        x = self.embedding(x)
        B, T, _ = x.shape
        
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        return x
    
    def forward(self, x):
        # Preprocess input
        #x = img_to_patch(x, self.patch_size)
        x = self.embedding(x)
        B, T, _ = x.shape
        #x = self.input_layer(x)

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
        
class VisionDecoder(nn.Module):
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
        super().__init__()

        self.patch_size = patch_size
        # Layers/Networks
        self.transformer = nn.Sequential(
            *(CrossAttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.ffn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
    
    def forward(self, x):
        # Preprocess input
        B, T, _ = x.shape

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

class ViT_trans(BaseLightningClass):
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
        self.f1_cal = F1Score(num_classes=model_kwargs['num_classes'], task = 'multiclass')

    def forward(self, x):
        if isinstance(x, list) and all(isinstance(item, torch.Tensor) for item in x):
            x = torch.stack(x).permute(1, 0, 2, 3, 4)
            
        embeddings_list = []
        for imgs in x:
            embeddings = self.model.get_embedding(imgs)  # shape (4, embedding_size)
            embeddings_list.append(embeddings)
        #print("get_embeding의 임베딩 모양",embeddings.shape)
        embeddings = torch.stack(embeddings_list, 0) # bsz, 4, embdding_size(이미지 하나의 vit cls token representation dim)
        #print("stack을 쌓은 뒤 임베딩 모양",embeddings.shape)
        ###############################
        B, _, _ = embeddings.shape
        cls_token = self.cls_token.repeat(B, 1, 1) # [B, 1, embed_dim]
        #print("cls_token  모양",cls_token.shape)
        embeddings = torch.cat([cls_token, embeddings], dim=1) # [B, 5, embed_dim]
        #print("cls토큰이 결합 된 뒤 embeding 모양",embeddings.shape)
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


class ViT_QA_cos(BaseLightningClass):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = VisionTransformer(**model_kwargs)
        ##############################
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_kwargs['embed_dim']))
        self.pos_embedding = nn.Parameter(torch.randn(1, 5, model_kwargs['embed_dim']))
        self.dropout = nn.Dropout(model_kwargs['dropout'])
        '''
        self.Crosstransformer = nn.Sequential(
            *(CrossAttentionBlock(model_kwargs['embed_dim'], model_kwargs['hidden_dim'], model_kwargs['num_heads'], dropout=model_kwargs['dropout']) for _ in range(model_kwargs['head_num_layers']))
        )
        '''
        ##############################
        self.mlp_head = nn.Sequential(nn.LayerNorm(model_kwargs['embed_dim']),
                                      nn.Linear(model_kwargs['embed_dim'], model_kwargs['num_classes'])
                                      )
        self.f1_cal = F1Score(num_classes=model_kwargs['num_classes'], task = 'multiclass')

    def forward(self, x):
        B, _,_, _ ,_= x.shape
        x = x.permute(1, 0, 2, 3, 4)
        #print(x.shape)
        cls_list = []
        for imgs in (x):
            embeddings = self.encoder.get_embedding(imgs)  # shape (4, embedding_size)
            #print(embeddings.shape)
            cls_list.append(embeddings)
            #print("임베딩 모양",embeddings.shape)
        cls = torch.stack(cls_list,0)
        #print("cls 차원(append된것과 차이 없음): ",cls.shape)
        ##################원래 mlp##############
        #preds = self.mlp_head(embeddings) # bsz, num_classes
        ##################코사인 유사도##############
        # 0번째 시퀀스와 나머지 시퀀스 분리
        query_seq = cls[0].unsqueeze(0) # (1, batch, embedding)
        candidate_seqs = cls[1:]        # (3, batch, embedding)

        # 코사인 유사도 계산
        cos_similarities = F.cosine_similarity(query_seq, candidate_seqs, dim=2)

        cos_similarities = cos_similarities.transpose(1,0)
        #print(preds)
        
        return cos_similarities
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)
        return [optimizer], [lr_scheduler]

class ViT_QA2(BaseLightningClass):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = VisionTransformer(**model_kwargs)
        ##############################
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_kwargs['embed_dim']))
        self.pos_embedding = nn.Parameter(torch.randn(1, 5, model_kwargs['embed_dim']))
        self.dropout = nn.Dropout(model_kwargs['dropout'])
        '''
        self.Crosstransformer = nn.Sequential(
            *(CrossAttentionBlock(model_kwargs['embed_dim'], model_kwargs['hidden_dim'], model_kwargs['num_heads'], dropout=model_kwargs['dropout']) for _ in range(model_kwargs['head_num_layers']))
        )
        '''
        self.decoder = VisionDecoder(**model_kwargs)
        ##############################
        self.mlp_head = nn.Sequential(nn.LayerNorm(model_kwargs['embed_dim']),
                                      nn.Linear(model_kwargs['embed_dim'], model_kwargs['num_classes'])
                                      )
        self.f1_cal = F1Score(num_classes=model_kwargs['num_classes'], task = 'multiclass')

    def forward(self, x):
        B, _,_, _ ,_= x.shape
        x = x.permute(1, 0, 2, 3, 4)
        #print(x.shape)
        embeddings_list = []
        cls_list = []
        for imgs in (x):
            embeddings = self.encoder.get_value(imgs)  # shape (4, embedding_size)
            #print(embeddings.shape)
            embeddings_list.append(embeddings)
            cls_list.append(embeddings[0])
            #print("임베딩 모양",embeddings.shape)
        embeddings = torch.stack(embeddings_list, 0) # bsz, 4, embdding_size(이미지 하나의 vit cls token representation dim)
        #print("임베딩 차원(append된것과 차이 없음): ",embeddings.shape)
        
        cls = torch.stack(cls_list,0)
        #print("cls 차원(append된것과 차이 없음): ",cls.shape)
        ##################원래 mlp##############
        #preds = self.mlp_head(embeddings) # bsz, num_classes
        ##################코사인 유사도##############
        # 0번째 시퀀스와 나머지 시퀀스 분리
        query_seq = cls[0].unsqueeze(0) # (1, batch, embedding)
        candidate_seqs = cls[1:]        # (3, batch, embedding)

        # 코사인 유사도 계산
        cos_similarities = F.cosine_similarity(query_seq, candidate_seqs, dim=2)

        cos_similarities = cos_similarities.transpose(1,0)
        #print(preds)
        
        return cos_similarities
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)
        return [optimizer], [lr_scheduler]
    
if __name__ == "__main__":
    
    img = torch.ones([16, 4, 3, 128, 128])
    print("Is x a list of tensors?", all(isinstance(item, torch.Tensor) for item in img))
    print("Length of x:", len(img))
    
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
    #model = ViT_trans(model_kwargs,lr=1e-3)
    model = ViT_QA_cos(model_kwargs,lr=1e-3)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]