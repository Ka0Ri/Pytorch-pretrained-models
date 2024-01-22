import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torchmetrics.classification import Accuracy, Dice, F1Score
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Union, Tuple, List, Dict
from torchmetrics import MetricCollection
import torch.nn.functional as F
from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks
import cv2
from pytorch_lightning.loggers import NeptuneLogger
from PIL import Image

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
   
# For training

def get_lr_scheduler_config(monitor: str,
                            lr_scheduler: str='step',
                            **kwargs) -> Dict[str, Union[optim.lr_scheduler._LRScheduler, str, str, int]]:
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
        'step': lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, **kwargs),
        'multistep': lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=10, **kwargs),
        'reduce_on_plateau': lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    }

    scheduler_creator = scheduler_mapping.get(lr_scheduler)

    if scheduler_creator is not None:
        scheduler = scheduler_creator
    else:
        raise NotImplementedError

    return {
        'scheduler': scheduler,
        'monitor': monitor,
        'interval': 'epoch',
        'frequency': 1,
    }


class CustomMAP(MeanAveragePrecision):
    '''
    Customized MeanAveragePrecision for pytorch-lightning
    '''
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        '''
        Update metric
        Args:
            preds: predicted output
            target: ground truth
        '''
        preds = [{k: v.cpu() for k, v in t.items()} for t in preds]
        target = [{k: v.cpu() for k, v in t.items()} for t in target]
        if 'scores' not in preds[0].keys():
            preds = [{**t, 'scores': torch.ones_like(t['labels'])} for t in preds]

        super().update(preds, target)

    def compute(self):
        '''
        Compute metric
        Returns:
            metric: metric value mAP
        '''
        metric = super().compute()
      
        return metric['map']

    
def get_metrics(metric_names: list[str], num_classes: int, prefix: str) -> Dict[str, Union[Accuracy, CustomMAP, Dice]]:
    """
    Set up metrics for evaluation
    Args:
        metric_names: list of metric names
        num_classes: number of classes for relevant metrics
    Returns:
        metrics: dictionary of metric instances
    """
    metric_mapping = {
        'accuracy': lambda: Accuracy(task='multiclass', num_classes=num_classes),
        'f1': lambda: F1Score(task='multiclass', num_classes=num_classes, average='macro'),
        'map': CustomMAP,
        'dice': lambda: Dice(num_classes=num_classes),
    }

    metrics = {}

    for metric_name in metric_names:
        metric_creator = metric_mapping.get(metric_name)

        if metric_creator is not None:
            metrics[metric_name] = metric_creator()
        else:
            raise NotImplementedError(f"Metric '{metric_name}' is not implemented.")

    return MetricCollection(metrics, prefix=prefix)


def get_optimizer(  optimizer: str='adam',
                    lr: int=0.0001,
                    **kwargs) -> optim.Optimizer:
    """
    Set up learning optimizer
    Args:
        parameters: model's parameters
        settings: settings hyperparameters
    Returns:
        optimizer: optimizer
    """
    optimizer_mapping = {
        'adam': lambda params: optim.Adam(params, lr=lr, **kwargs),
        'sgd': lambda params: optim.SGD(params, lr=lr, **kwargs),
    }

    optimizer_creator = optimizer_mapping.get(optimizer)

    if optimizer_creator is not None:
        return optimizer_creator
    else:
        raise NotImplementedError()

class NoLoss(nn.Module):
    '''
    Customized loss function for pytorch-lightning
    '''
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        '''
        Compute loss
        Args:
            y_hat: predicted output
            y: ground truth
        Returns:
            loss: loss value
        '''
        
        return -1

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
        'none': NoLoss(),  # Only for task == detection
    }

    loss_function = loss_mapping.get(loss_type)

    if loss_function is not None:
        return loss_function
    else:
        raise NotImplementedError()

def get_gpu_settings(gpu_ids: list[int]) -> Tuple[str, int, str]:
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

    n_gpu = len(gpu_ids)
    mapping = {
        'devices': gpu_ids if gpu_ids is not None else n_gpu if n_gpu is not None else 1,
        'strategy': 'ddp' if (gpu_ids or n_gpu) and (len(gpu_ids) > 1 or n_gpu > 1) else 'auto'
    }

    return "gpu", mapping['devices'], mapping['strategy']


def get_basic_callbacks(mode: str,
                        monitor: str,
                        ckpt_path: str, 
                        early_stopping: bool = False,
                        **kwargs) -> List[Union[LearningRateMonitor, ModelCheckpoint, EarlyStopping]]:
    '''
    Get basic callbacks for PyTorch Lightning Trainer.
    Args:
        metric_name: name of the metric
        ckpt_path: path to save the checkpoints
        early_stopping: flag for early stopping callback
    Returns:
        callbacks: list of callbacks
    '''
 
    callbacks_mapping = {
        'last': ModelCheckpoint(dirpath=ckpt_path, filename='{epoch:03d}', monitor=None, **kwargs),
        'best': ModelCheckpoint(dirpath=ckpt_path, filename='{epoch:03d}', monitor=monitor, mode=mode, **kwargs),
        'lr': LearningRateMonitor(logging_interval='epoch', **kwargs),
        'early_stopping': EarlyStopping(monitor=monitor, mode=mode, **kwargs),
    }

    callbacks = [callbacks_mapping[key] for key in ['last', 'best', 'lr']]
    
    if early_stopping:
        callbacks.append(callbacks_mapping['early_stopping'])

    return callbacks
    
def get_trainer(logger,
                gpu_ids: list[int],
                monitor: str,
                ckpt_path: str,
                mode: str,
                max_epochs: int=10,
                early_stopping: bool=False,
                **kwargs) -> Trainer:
    '''
    Get trainer and logging for pytorch-lightning trainer:
    Args: 
        settings: hyperparameter settings
        task: task to run training
    Returns:
        trainer: trainer object
        logger: neptune logger object
    '''
    callbacks = get_basic_callbacks(mode, monitor, ckpt_path, early_stopping, **kwargs)
    accelerator, devices, strategy = get_gpu_settings(gpu_ids)

    return Trainer(
        logger=[logger],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
    )

def get_masks_overlay(logger: NeptuneLogger,
                       phase: str,
                       images: torch.Tensor,
                       predictions: torch.Tensor,
                       targets: torch.Tensor=None,
                       class_names: List[str]=None,
                       num_of_images: int=16,
                       threshold: float=0.5) -> torch.Tensor:
    '''
    Get bounding boxes overlay for images
    Args:
        images: images
        targets: ground truth bounding boxes
        predictions: predicted bounding boxes
        class_names: list of class names
        threshold: threshold for IoU
    Returns:
        images: images with bounding boxes overlay
    '''
    n = min(num_of_images, len(images))

    predictions = [{k: v.cpu() for k, v in t.items()} for t in predictions[:n]]
    images = images[:n]
    # images = [(t * 255).to(torch.uint8) for t in images[:n]]

    boolean_masks = [out['masks'][out['scores']  > .75] > threshold for out in predictions]
    reconstructions = [draw_segmentation_masks(image, mask.squeeze(1), alpha=0.9) 
                            for image, mask in zip(images, boolean_masks)]
    
    reconstructions = torch.stack([F.interpolate(img.unsqueeze(0), size=(128, 128))
                                    for img in reconstructions]).squeeze(1) 
    reconstructions = make_grid(reconstructions, nrow= int(n ** 0.5))
    reconstructions = reconstructions.numpy().transpose(1, 2, 0) / 255

    logger.experiment[phase].append(Image.fromarray(reconstructions))

def get_bboxes_overlay(logger: NeptuneLogger,
                      phase: str,
                      images: torch.Tensor,
                      predictions: torch.Tensor,
                      targets: torch.Tensor=None,
                      class_names: List[str]=None,
                      num_of_images: int=4,
                      threshold: float=0.5) -> torch.Tensor:
    '''
    Get masks overlay for images
    Args:
        images: images
        targets: ground truth masks
        predictions: predicted masks
        class_names: list of class names
        threshold: threshold for IoU
    Returns:
        images: images with masks overlay
    '''

    n = min(num_of_images, len(images))

    predictions = [{k: v.cpu() for k, v in t.items()} for t in predictions[:n]]
    images = [t.cpu() for t in images[:n]]

    boxes = [out['boxes'][out['scores'] > threshold] for out in predictions]
    reconstructions = [draw_bounding_boxes(image, box, width=4, colors='red')
                                    for image, box in zip(images, boxes)]
    
    reconstructions = torch.stack([F.interpolate(img.unsqueeze(0), size=(128, 128))
                                    for img in reconstructions]).squeeze(1) 
    reconstructions = make_grid(reconstructions, nrow= int(n ** 0.5))
    reconstructions = reconstructions.numpy().transpose(1, 2, 0)

    logger.experiment[phase].append(Image.fromarray(reconstructions))
 

def get_class_overlay(
                    logger: NeptuneLogger,
                    phase: str,
                    images: torch.Tensor,
                    predictions: torch.Tensor,
                    targets: torch.Tensor=None,
                    class_names: List[str]=None,
                    num_of_images: int=16,
                    **kwargs) -> torch.Tensor:
    '''
    Get class overlay for images
    Args:
        images: images
        targets: ground truth class
        predictions: predicted class
        class_names: list of class names
    Returns:
        images: images with class overlay
    '''
    
    images = images.clone().detach().cpu()
    predictions = predictions.clone().detach().cpu()
    n = min(num_of_images, images.shape[0])

    predictions = torch.argmax(predictions, dim=-1)
    images = (images.numpy() * 255).astype(np.uint8)
   
    images = [cv2.resize(image, (128, 128)) for image in images]
    
    draw_images = np.array([cv2.putText(image, str(label.item()) if class_names is None else class_names[int(label.item())], 
                                        (64, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                                    for image, label in zip(images[:n], predictions[:n])])

    reconstructions = torch.from_numpy(draw_images.transpose(0, 3, 1, 2))
    reconstructions = make_grid(reconstructions, nrow= int(n ** 0.5))
    reconstructions = reconstructions.numpy().transpose(1, 2, 0)

    logger.experiment[phase].append(Image.fromarray(reconstructions))

