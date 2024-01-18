import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torchmetrics.classification import Accuracy, Dice
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Union, Tuple, List, Dict

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
   
# For training

def get_lr_scheduler_config(optimizer: optim.Optimizer,
                            metric_name: str,
                            lr_scheduler: str='step',
                            lr_step: int=10,
                            lr_decay: float=0.8,
                            frequency: int=1) -> Dict[str, Union[optim.lr_scheduler._LRScheduler, str, str, int]]:
    '''
    Set up learning rate scheduler configuration.
    Args:
        optimizer: optimizer
        lr_scheduler: type of learning rate scheduler
        lr_step: step size for scheduler
        lr_decay: decay factor for scheduler
        metric: metric for scheduler monitoring
    Returns:
        lr_scheduler_config: learning rate scheduler configuration
    '''
    scheduler_mapping = {
        'step': lambda: torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay),
        'multistep': lambda: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=lr_decay),
        'reduce_on_plateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001)
    }

    scheduler_creator = scheduler_mapping.get(lr_scheduler)

    if scheduler_creator is not None:
        scheduler = scheduler_creator()
    else:
        raise NotImplementedError

    return {
        'scheduler': scheduler,
        'monitor': f'metrics/batch/val_{metric_name}',
        'interval': 'epoch',
        'frequency': frequency,
    }


class CustomMAP(MeanAveragePrecision):
    '''
    Customized MeanAveragePrecision for pytorch-lightning
    '''
    def __init__(self):
        super().__init__()

    def compute(self):
        '''
        Compute metric
        Returns:
            metric: metric value mAP
        '''
        metric = super().compute()
      
        return metric['map']

def get_metric(metric_name: str, 
               num_classes: int) -> Union[Accuracy, CustomMAP, Dice]:
    """
    Set up metric for evaluation
    Args:
        metric_name: name of metric
        num_classes: number of classes for relevant metrics
    Returns:
        metric: metric function
    """
    metric_mapping = {
        'acc': lambda: Accuracy(num_classes=num_classes),
        'mAP': CustomMAP,
        'dice': lambda: Dice(num_classes=num_classes),
    }

    metric_creator = metric_mapping.get(metric_name)

    if metric_creator is not None:
        return metric_creator()
    else:
        raise NotImplementedError()

def get_optimizer(parameters: List[nn.Parameter],
                    optimizer: str='adam',
                    lr: int=0.0001,
                    weight_decay: float=0.005,
                    momentum: float=0.9) -> optim.Optimizer:
    """
    Set up learning optimizer
    Args:
        parameters: model's parameters
        settings: settings hyperparameters
    Returns:
        optimizer: optimizer
    """
    optimizer_mapping = {
        'adam': lambda: optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
        'sgd': lambda: optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum),
    }

    optimizer_creator = optimizer_mapping.get(optimizer)

    if optimizer_creator is not None:
        return optimizer_creator()
    else:
        raise NotImplementedError()

def get_loss_function(loss_type: str) -> nn.Module:
    """
    Set up loss function
    Args:
        loss_type: loss function type
    Returns:
        loss: loss function
    """
    loss_mapping = {
        'ce': nn.CrossEntropyLoss(),
        'bce': nn.BCELoss(),
        'mse': nn.MSELoss(),
        'none': None,  # Only for task == detection
    }

    loss_function = loss_mapping.get(loss_type)

    if loss_function is not None or loss_type == 'none':
        return loss_function
    else:
        raise NotImplementedError()

def get_gpu_settings(gpu_ids: list[int],
                     n_gpu: int) -> Tuple[str, int, str]:
    '''
    Get GPU settings for PyTorch Lightning Trainer:
    Args:
        gpu_ids (list[int])
        n_gpu (int)
    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    '''
    if not torch.cuda.is_available():
        return "cpu", None, None

    mapping = {
        'devices': gpu_ids if gpu_ids is not None else n_gpu if n_gpu is not None else 1,
        'strategy': 'ddp' if (gpu_ids or n_gpu) and (len(gpu_ids) > 1 or n_gpu > 1) else 'auto'
    }

    return "gpu", mapping['devices'], mapping['strategy']


def get_basic_callbacks(metric_name: str, 
                        ckpt_path: str, 
                        early_stopping: bool = False) -> List[Union[LearningRateMonitor, ModelCheckpoint, EarlyStopping]]:
    '''
    Get basic callbacks for PyTorch Lightning Trainer.
    Args:
        metric_name: name of the metric
        ckpt_path: path to save the checkpoints
        early_stopping: flag for early stopping callback
    Returns:
        callbacks: list of callbacks
    '''
    common_params = {
        'dirpath': ckpt_path,
        'filename': '{epoch:03d}',
        'auto_insert_metric_name': False,
        'save_top_k': 1,
    }
    mode = 'min' if metric_name == 'loss' else 'max'

    callbacks_mapping = {
        'last': ModelCheckpoint(**common_params, monitor=None),
        'best': ModelCheckpoint(**common_params, monitor=f'metrics/epoch/val_{metric_name}', mode=mode, verbose=True),
        'lr': LearningRateMonitor(logging_interval='epoch'),
        'early_stopping': EarlyStopping(monitor=f'metrics/epoch/val_{metric_name}', mode=mode, patience=10),
    }

    callbacks = [callbacks_mapping[key] for key in ['last', 'best', 'lr']]
    
    if early_stopping:
        callbacks.append(callbacks_mapping['early_stopping'])

    return callbacks
    
def get_trainer(logger,
                gpu_ids: list[int],
                n_gpu: int,
                metric_name: str,
                ckpt_path: str,
                early_stopping: bool=False,
                max_epochs: int=10) -> Trainer:
    '''
    Get trainer and logging for pytorch-lightning trainer:
    Args: 
        settings: hyperparameter settings
        task: task to run training
    Returns:
        trainer: trainer object
        logger: neptune logger object
    '''
    callbacks = get_basic_callbacks(metric_name, ckpt_path, early_stopping)
    accelerator, devices, strategy = get_gpu_settings(gpu_ids, n_gpu)

    trainer = Trainer(
        logger=[logger],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
    )
    return trainer

