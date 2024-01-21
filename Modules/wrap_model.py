from pytorch_lightning import LightningModule
import Modules.Model.base_model as base_model
from typing import Union
from torchmetrics import MetricCollection
import torch.nn as nn


class WrapModel(LightningModule):
    '''
    Wrapping model with in lightning module
    '''
    def __init__(self, 
                model: base_model,
                optimizer_fn: None,
                loss_fn: Union[nn.Module, None]=None,
                train_metrics_fn: MetricCollection=None,
                val_metrics_fn: MetricCollection=None,
                test_metrics_fn: MetricCollection=None,
                lr_scheduler_fn=None,
                log_fn=None):
        
        super().__init__()
      
        # Model selection
        self.model = model
        self.optimizer_fn = optimizer_fn
        self.lr_scheduler_fn = lr_scheduler_fn
        self.loss_fn = loss_fn
        self.train_metrics_fn = train_metrics_fn
        self.val_metrics_fn = val_metrics_fn
        self.test_metrics_fn = test_metrics_fn
        self.log_fn = log_fn

    def setup(self, stage: str):

        if stage == "fit":
            # Loss selection
            assert self.val_metrics_fn is not None, "Validation metrics is not defined"
            assert self.train_metrics_fn is not None, "Training metrics is not defined"
            assert self.loss_fn is not None, "Loss function is not defined"
            assert self.optimizer_fn is not None, "Optimizer is not defined"

        elif stage == "test":
            assert self.test_metrics_fn is not None, "Test metrics is not defined"
            
    def forward(self, x, y=None):

        return self.model(x, y)
    
    def on_train_epoch_start(self) -> None:

        self.train_output_list = []
        return super().on_train_epoch_start()
     
    def training_step(self, batch, batch_idx):

        x, y = batch['data'], batch['target']
       
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        #log metrics
        self.train_metrics_fn(y_hat, y)
      
    def on_train_epoch_end(self):

        self.log_dict(self.train_metrics_fn, prog_bar=True)
       
    def on_validation_epoch_start(self) -> None:
        self.val_output_list = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):

        x, y = batch['data'], batch['target']

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=False)
        # log metrics
        self.val_metrics_fn(y_hat, y)
        self.val_output_list.append({'original': batch['original'], 'predictions': y_hat, 'targets': y})
                                     
    def on_validation_epoch_end(self):

        self.log_dict(self.val_metrics_fn, prog_bar=True)
        # convert torch tensor to PIL image
        if self.log_fn is not None:
            self.log_fn(self.logger, phase='val_outputs', 
                        images=self.val_output_list[0]['original'],
                        predictions=self.val_output_list[0]['predictions'],
                        targets=self.val_output_list[0]['targets'])
       
       
    def on_test_epoch_start(self) -> None:
        self.test_output_list = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):

        x, y = batch['data'], batch['target']
        y_hat = self(x)
        self.test_metrics_fn(y_hat, y)
        self.test_output_list.append({'original': batch['original'], 'predictions': y_hat, 'targets': y})
    
    def on_test_epoch_end(self):

        self.log_dict(self.test_metrics_fn, prog_bar=True)
        if self.log_fn is not None:
            self.log_fn(self.logger, phase='test_outputs',
                        images=self.test_output_list[0]['original'],
                        predictions=self.test_output_list[0]['predictions'],
                        targets=self.test_output_list[0]['targets'])
             
    def configure_optimizers(self):

        optimizer = self.optimizer_fn(self.parameters())
        if(self.lr_scheduler_fn is not None):
            return {"optimizer": optimizer, "lr_scheduler": self.lr_scheduler_fn}
        
        return {"optimizer": optimizer}
