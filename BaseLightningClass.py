import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import f1_score
from torchmetrics import F1Score
import torch
import wandb

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

        # 에폭당 로그 기록 간격 설정
        if self.trainer.num_training_batches is not None:
            log_interval = max(1, self.trainer.num_training_batches // 10)
            if (self.global_step + 1) % log_interval == 0:
                self.log("%s_loss" % mode, loss, on_step=True, on_epoch=True)
                self.log("%s_acc" % mode, acc, on_step=True, on_epoch=True)
                self.log("%s_f1" % mode, f1, on_step=True, on_epoch=True)

        return loss, {"preds": pred_class, "gts": labels, "categories": categories}


    def training_step(self, batch, batch_idx):

        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        _, result = self._calculate_loss(batch, mode="val")
        # result 딕셔너리에서 필요한 값을 올바르게 추출
        # 예: loss 값 추출
        val_loss = result["loss"]
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        return result
    def test_step(self, batch, batch_idx):
        _, result = self._calculate_loss(batch, mode="test")
        # 검증 손실 로그 기록
        test_loss = result['loss']
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)
        return result

    def predict_step(self, batch, batch_idx):
        imgs_list, labels, categories = batch

        labels -= 1
        preds = self(imgs_list)        
        pred_class = preds.argmax(dim=-1)
        

        return {"preds": pred_class, "gts": labels, "categories": categories}

    def on_after_backward(self):
        if self.global_step % 25 == 0:  # 예를 들어, 매 25 스텝마다 로그
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # 그래디언트 추적 제거 및 CPU로 이동
                    grad = param.grad.detach().cpu()
                    weight = param.detach().cpu()
                    # wandb에 로그 기록
                    self.logger.experiment.log({f"train/{name}_grad": wandb.Histogram(grad.numpy())})
                    self.logger.experiment.log({f"train/{name}_weight": wandb.Histogram(weight.numpy())})
