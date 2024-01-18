import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import f1_score
from torchmetrics import F1Score
import torch

class BaseLightningClass(pl.LightningModule):

    def _calculate_loss(self, batch, mode="train"):
        imgs_list, labels, categories = batch
        labels = torch.squeeze(labels)
        labels -= 1
        preds = self(imgs_list)
        loss = F.cross_entropy(preds, labels)

        f1 = f1_score(preds, labels, num_classes=3, task='multiclass')
        
        pred_class = preds.argmax(dim=-1)
        acc = (pred_class == labels).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        self.log("%s_f1" % mode, f1)
        return loss, {"preds": pred_class, "gts": labels, "categories": categories}

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
