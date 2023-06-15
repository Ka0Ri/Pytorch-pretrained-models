import numpy as np
import torch
import random
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.utils import make_grid, draw_bounding_boxes

from Modules.Classifier import WrappingClassifier
from Modules.Segment import WrappingSegment
from Modules.Detector import WrappingDetector
from Modules.ultis import *

from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from neptune.types import File
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision


seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_float32_matmul_precision('medium')

def get_lr_scheduler_config(optimizer, settings):
    '''
    set up learning rate scheduler
    Args:
        optimizer: optimizer
        settings: settings hyperparameters
    Returns:
        lr_scheduler_config: [learning rate scheduler, configuration]
    '''
    if settings['lr_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=settings['lr_step'], gamma=settings['lr_decay'])
    elif settings['lr_scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=settings['lr_step'], gamma=settings['lr_decay'])
    elif settings['lr_scheduler'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001)
    else:
        raise NotImplementedError

    return {
            'scheduler': scheduler,
            'monitor': 'metrics/batch/train_loss',
            'interval': 'epoch',
            'frequency': 1,
        }

def get_optimizer(parameters, settings):
    '''
    set up learning optimizer
    Args:
        parameters: model's parameters
        settings: settings hyperparameters
    Returns:
        optimizer: optimizer
    '''
    if settings['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=settings['lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            parameters, lr=settings['lr'], weight_decay=settings['weight_decay'], momentum=settings['momentum'])
    else:
        raise NotImplementedError()

    return optimizer

def get_loss_function(type):
    '''
    set up loss function
    Args:
        settings: settings hyperparameters,
    Returns:
        loss: loss function
    '''
    if type == "ce": 
        loss = nn.CrossEntropyLoss()
    elif type == "nll": 
        loss = nn.NLLLoss()
    elif type == "bce": 
        loss = nn.BCELoss()
    elif type == "mse": 
        loss = nn.MSELoss()
    elif type == "none": 
        loss = None # only for task == detection
    else: 
        raise NotImplementedError()

    return loss

def get_gpu_settings(gpu_ids, n_gpu):
    '''
    Get gpu settings for pytorch-lightning trainer:
    Args:
        gpu_ids (list[int])
        n_gpu (int)
    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    '''
    if not torch.cuda.is_available():
        return "cpu", None, None

    if gpu_ids is not None:
        devices = gpu_ids
        strategy = "ddp" if len(gpu_ids) > 1 else 'auto'
    elif n_gpu is not None:
        devices = n_gpu
        strategy = "ddp" if n_gpu > 1 else 'auto'
    else:
        devices = 1
        strategy = 'auto'

    return "gpu", devices, strategy

def get_basic_callbacks(settings):
    '''
    Get basic callbacks for pytorch-lightning trainer:
    Args: 
        settings
    Returns:
        last ckpt, best ckpt, lr callback, early stopping callback
    '''
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    last_ckpt_callback = ModelCheckpoint(
        filename='last_model_{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=None,
    )
    best_ckpt_calllback = ModelCheckpoint(
        filename='best_model_{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor='metrics/batch/train_loss',
        mode='min',
        verbose=True
    )
    if settings['early_stopping']:
        early_stopping_callback = EarlyStopping(
            monitor='metrics/batch/train_loss',  # Metric to monitor for improvement
            mode='min',  # Choose 'min' or 'max' depending on the metric (e.g., 'min' for loss, 'max' for accuracy)
            patience=10,  # Number of epochs with no improvement before stopping
        )
        return [last_ckpt_callback, best_ckpt_calllback, lr_callback, early_stopping_callback]
    else: 
        return [last_ckpt_callback, best_ckpt_calllback, lr_callback]

def get_trainer(settings, logger) -> Trainer:
    '''
    Get trainer and logging for pytorch-lightning trainer:
    Args: 
        settings: hyperparameter settings
        task: task to run training
    Returns:
        trainer: trainer object
        logger: neptune logger object
    '''
    callbacks = get_basic_callbacks(settings)
    accelerator, devices, strategy = get_gpu_settings(settings['gpu_ids'], settings['n_gpu'])

    trainer = Trainer(
        logger=[logger],
        max_epochs=settings['n_epoch'],
        default_root_dir=settings['ckpt_path'],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
    )
    return trainer

class DataModule(LightningDataModule):
    '''
    Data Module for Train/Val/Test data loadding
    Args: 
        data_settings, training_settings: hyperparameter settings
    Returns:
        Train/Test/Val data loader
    '''
    def __init__(self, data_settings, training_settings, transform=[None, None]):
        super().__init__()

        self.dataset = data_settings['name']
        self.root_dir = data_settings['path']
        self.img_size = data_settings['img_size']
        self.batch_size = training_settings['n_batch']
        self.num_workers = training_settings['num_workers']
        self.train_transform, self.val_transform = transform
        self.class_list = None
        self.collate_fn = None

    def setup(self, stage: str):

        if stage == "fit":
            if self.dataset == 'CIFAR10':
                self.Train_dataset = CIFAR10read(mode="train", data_path=self.root_dir, 
                                                transform=self.train_transform, imgsize=self.img_size)
                self.Val_dataset =  CIFAR10read(mode="val", data_path=self.root_dir, 
                                                transform=self.val_transform, imgsize=self.img_size)
            elif self.dataset == 'LungCT-Scan':
                train_dataset = LungCTscan(data_path=self.root_dir, transform=self.train_transform, imgsize=self.img_size)
                self.Train_dataset = Subset(train_dataset, range(int(len(train_dataset) * 0.8)))
                val_dataset = LungCTscan(data_path=self.root_dir, transform=self.val_transform, imgsize=self.img_size)
                self.Val_dataset = Subset(val_dataset, range(int(len(val_dataset) * 0.8), len(val_dataset)))
            elif self.dataset == 'Dubai':
                train_dataset = DubaiAerialread(data_path=self.root_dir, transform=self.train_transform, imgsize=self.img_size)
                self.Train_dataset = Subset(train_dataset, range(int(len(train_dataset) * 0.8)))
                val_dataset = DubaiAerialread(data_path=self.root_dir, transform=self.val_transform, imgsize=self.img_size)
                self.Val_dataset = Subset(val_dataset, range(int(len(val_dataset) * 0.8), len(val_dataset)))
            elif self.dataset == 'PennFudan':
                self.collate_fn = collate_fn
                train_dataset = PennFudanDataset(data_path=self.root_dir, transform=self.train_transform, imgsize=self.img_size)
                self.Train_dataset = Subset(train_dataset, range(int(len(train_dataset) * 0.8)))
                val_dataset = PennFudanDataset(data_path=self.root_dir, transform=self.val_transform, imgsize=self.img_size)
                self.Val_dataset = Subset(val_dataset, range(int(len(val_dataset) * 0.8), len(val_dataset)))
                
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            if self.dataset == 'CIFAR10':
                self.Test_dataset =  CIFAR10read(mode="test", data_path=self.root_dir, 
                                                transform=self.val_transform, imgsize=self.img_size)
            elif self.dataset == 'LungCT-Scan':
                dataset = LungCTscan(data_path=self.root_dir, transform=self.val_transform, imgsize=self.img_size)
                self.Test_dataset = Subset(dataset, range(len(dataset)))
            elif self.dataset == 'Dubai':
                dataset = DubaiAerialread(data_path=self.root_dir, transform=self.val_transform, imgsize=self.img_size)
                self.Test_dataset = Subset(dataset, range(len(dataset)))
            elif self.dataset == 'PennFudan':
                self.collate_fn = collate_fn
                dataset = PennFudanDataset(data_path=self.root_dir, transform=self.val_transform, imgsize=self.img_size)
                self.Test_dataset = Subset(dataset, range(len(dataset)))

    def train_dataloader(self):
        return DataLoader(self.Train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.Val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.Test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
    
class Model(LightningModule):
    def __init__(self, PARAMS, task=None):
        super().__init__()
        self.save_hyperparameters()

        self.architect_settings = PARAMS['architect_settings']
        self.train_settings = PARAMS['training_settings']
        self.dataset_settings = PARAMS['dataset_settings']
        self.task = task
        # Model selection
        if(self.task == 'classification'):
            self.model = WrappingClassifier(model_configs=self.architect_settings)
            self.train_metrics = torchmetrics.Accuracy(task='multiclass', num_classes=self.architect_settings['n_cls'])
            self.valid_metrics = torchmetrics.Accuracy(task='multiclass', num_classes=self.architect_settings['n_cls'])
        elif(self.task == 'segmentation'):
            self.model = WrappingSegment(model_configs=self.architect_settings)
            self.train_metrics = torchmetrics.Dice(num_classes=self.architect_settings['n_cls'])
            self.valid_metrics = torchmetrics.Dice(num_classes=self.architect_settings['n_cls'])
        elif(self.task == 'detection'):
            self.model = WrappingDetector(model_configs=self.architect_settings)
            self.train_metrics = MeanAveragePrecision()
            self.valid_metrics = MeanAveragePrecision()
        else:
            raise NotImplementedError()

        # Loss selection
        self.loss = get_loss_function(self.train_settings['loss'])
      
        self.validation_step_outputs = []
    
    def forward(self, x, y=None):
        return self.model(x, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if(self.task == 'detection'):
            loss_dict = self(x, y)
            loss = sum(loss for loss in loss_dict.values())
        else:
            y_hat = self(x)
            loss = self.loss(y_hat, y.long())
            y_pred = torch.softmax(y_hat, dim=1)
            self.train_metrics.update(y_pred.cpu(), y.cpu().long())

        self.log("metrics/batch/train_loss", loss, prog_bar=False)

        return loss

    def on_train_epoch_end(self):
       
        if(self.task == 'classification'):
            self.log("metrics/epoch/train_acc", self.train_metrics.compute())
        elif(self.task == 'segmentation'):
            self.log("metrics/epoch/train_dice", self.train_metrics.compute())
    
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if(self.task == 'detection'):
            y_hat = self(x)
            y_pred = [{k: v for k, v in t.items()} for t in y_hat]
            targets = [{k: v for k, v in t.items()} for t in y]

            self.valid_metrics.update(y_pred, targets)
            self.validation_step_outputs.append({"image": x[0], "predictions": y_pred[0], "targets": targets[0]})
        else:
            y_hat = self(x)
            loss = self.loss(y_hat, y.long())
            y_pred = torch.softmax(y_hat, dim=-1)
            self.valid_metrics.update(y_pred.cpu(), y.cpu().long())
        
            if(self.task == 'segmentation'):
                y_pred = torch.argmax(y_hat, dim=1)
                self.validation_step_outputs.append({"loss": loss.item(), "predictions": y_pred.unsqueeze(1), "targets": y.unsqueeze(1)})
            else:
                self.validation_step_outputs.append({"loss": loss.item()})

            self.log('metrics/batch/val_loss', loss)

    def on_validation_epoch_end(self):
        
        if(self.task == 'classification'):
            self.log('metrics/epoch/val_acc', self.valid_metrics.compute())
            loss =[outputs['loss'] for outputs in self.validation_step_outputs]
            self.log('metrics/epoch/val_loss', sum(loss) / len(loss))
           
        elif(self.task == 'segmentation'):
            self.log("metrics/epoch/val_dice", self.valid_metrics.compute())
            loss =[outputs['loss'] for outputs in self.validation_step_outputs]
            self.log('metrics/epoch/val_loss', sum(loss) / len(loss))

            outputs = self.validation_step_outputs
            reconstructions = make_grid(outputs[0]["predictions"], nrow=int(self.train_settings["n_batch"] ** 0.5))
            reconstructions = normalize_image(reconstructions.cpu().numpy().transpose(1, 2, 0))
            self.logger.experiment["val/reconstructions"].append(File.as_image(reconstructions))

            targets = make_grid(outputs[0]["targets"], nrow=int(self.train_settings["n_batch"] ** 0.5))
            targets = normalize_image(targets.cpu().numpy().transpose(1, 2, 0))
            self.logger.experiment["val/targets"].append(File.as_image(targets))
            self.validation_step_outputs.clear()
        
        elif(self.task == 'detection'):
            self.log('metrics/epoch/val_mAP', self.valid_metrics.compute()['map'])
            #no validation loss

            outputs = self.validation_step_outputs[-1]
            image, predictions, targets = outputs["image"], outputs["predictions"], outputs["targets"]
            reconstructions = draw_bounding_boxes((image * 255.).to(torch.uint8), 
                                                boxes=predictions["boxes"][:5],
                                                colors="red",
                                                width=5, font_size=20)
            reconstructions = draw_bounding_boxes(reconstructions, 
                                                boxes=targets["boxes"][:5],
                                                colors="blue",
                                                width=5, font_size=20)
            reconstructions = reconstructions.cpu().numpy().transpose(1, 2, 0) / 255.
            self.logger.experiment["val/reconstructions"].append(File.as_image(reconstructions))
            self.validation_step_outputs.clear()

        self.valid_metrics.reset()
       
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.train_settings)
        lr_scheduler_config = get_lr_scheduler_config(optimizer, self.train_settings)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}