from argparse import Namespace

import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
import timm
import pytorch_lightning as pl
import torch.nn.functional as F


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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  patience=self.hparams.reduce_lr_on_pleteau_patience,
                                                                  verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)
