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
    def on_train_epoch_end(self, outputs):
        # 훈련 에포크가 끝날 때 수행할 작업
        # 예: 평균 손실 계산
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_epoch_loss', avg_loss)

    def on_validation_epoch_end(self, outputs):
        # 검증 에포크가 끝날 때 수행할 작업
        # 예: 평균 정확도 계산
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_epoch_acc', avg_acc)

    def on_test_epoch_end(self, outputs):
        # 테스트 에포크가 끝날 때 수행할 작업
        # 예: 평균 F1 점수 계산
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        self.log('test_epoch_f1', avg_f1)