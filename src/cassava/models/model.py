from argparse import Namespace

import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
import timm
import pytorch_lightning as pl
import torch.nn.functional as F

from cassava.bitempered_loss import bi_tempered_logistic_loss


class LeafDoctorModel(pl.LightningModule):
    def __init__(self, hparams = None):
        super().__init__()
        self.hparams = hparams or Namespace()

        self.trunk = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)

        # for layer in [self.trunk.bn1, self.trunk.bn2]:
        #     for param in layer.parameters():
        #         param.requires_grad = False

    def forward(self, x):
        return self.trunk(x)

    def predict_proba(self, x):
        probabilities = nn.functional.softmax(self.forward(x), dim=1)
        return probabilities

    def predict(self, x):
        return torch.max(self.forward(x), 1)[1]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr or self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=self.hparams.lr,
                                                           epochs=self.hparams.max_epochs,
                                                           steps_per_epoch=int(23712/self.hparams.batch_size))
        return (
            [optimizer],
            [
                {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'val_loss',
                }
            ]
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = bi_tempered_logistic_loss(y_hat, y,
                                         self.hparams.bitempered_t1,
                                         self.hparams.bitempered_t2,
                                         label_smoothing=self.hparams.label_smoothing)
        acc = accuracy(y_hat, y)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)
