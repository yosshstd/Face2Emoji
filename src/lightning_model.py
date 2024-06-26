import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .models.vit import ViT


class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = ViT(**config.model)

        self.optimizer = config.train.optimizer
        self.vit_lr = config.train.vit_lr
        self.head_lr = config.train.head_lr
        self.weight_decay = config.train.weight_decay
        self.test_outputs = {'y_true': [], 'y_pred': []}
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        parameters = [
            {'params': self.model.vit.parameters(), 'lr': self.vit_lr},
            {'params': self.model.head.parameters(), 'lr': self.head_lr}
        ]

        if self.optimizer == 'Adam':
            opt = torch.optim.Adam(parameters, weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            opt = torch.optim.AdamW(parameters, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            patience=1.0,
            factor=0.8,
        )
        return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'val/loss'}
    
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        acc = (outputs.argmax(dim=-1) == y).float().mean()
        data_dict = {"loss": loss}
        log_dict = {"train/loss": loss, "train/acc": acc}
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        acc = (outputs.argmax(dim=-1) == y).float().mean()
        data_dict = {"loss": loss}
        log_dict = {"val/loss": loss, "val/acc": acc}
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs, y)
        acc = (outputs.argmax(dim=-1) == y).float().mean()
        data_dict = {"loss": loss}
        log_dict = {"test/loss": loss, "test/acc": acc}
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        
        self.test_outputs['y_true'].extend(y.detach().cpu().numpy())
        self.test_outputs['y_pred'].extend(outputs.argmax(dim=-1).detach().cpu().numpy())

        return data_dict


class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)
