import Modules.Model.Segment as Segment
import Modules.Model.Detector as Detector
import Modules.Model.Classifier as Classifier
from torchvision.transforms._presets import ObjectDetection
import os
from Modules.dataloader import base_dataset, DataModule
import numpy as np
import torch
from functools import partial
from PIL import Image
from Modules.dataloader import collate_fn_dict
from torchvision import datasets
from torchvision import transforms


class PennFudanDataset(base_dataset):

    def __init__(self, transform=None,
                 data_path='../../data/PennFudanPed/'):
        
        
        imgs = list(map(lambda img: os.path.join(data_path, "PNGImages", img), 
                        sorted(os.listdir(os.path.join(data_path, "PNGImages")))))
        masks = list(map(lambda mask: os.path.join(data_path, "PedMasks", mask), 
                        sorted(os.listdir(os.path.join(data_path, "PedMasks")))))
        
        if(transform is None):
            transform = partial(ObjectDetection)()

        super(PennFudanDataset, self).__init__(data=imgs, targets=masks, transform=transform)

    def __parse_mask__(self, mask):
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes, masks, obj_ids

    def __load_sample__(self, index) -> dict:
        
        img = Image.open(self.data[index]).convert("RGB")
        mask = Image.open(self.targets[index]).convert('L')
        original = img.copy()
        # from PIL to torch tensor
        original = torch.as_tensor(np.array(original), dtype=torch.uint8).permute(2, 0, 1)
        boxes, masks, obj_ids = self.__parse_mask__(mask)

        return {'data': img,
                'target': {'boxes': boxes,
                           'masks': masks,
                           'obj_ids': obj_ids},
                'original': original}
    
    def __transform__(self, sample: dict) -> dict:
        
        boxes = sample['target']['boxes']
        masks = sample['target']['masks']
        num_objs = len(sample['target']['obj_ids'])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([0])
        target["area"] = area
        target["iscrowd"] = iscrowd

        sample.update({'target': target})

        return super().__transform__(sample)
    

class CIFAR10(base_dataset):
    
    def __init__(self, data_path, train, transform=None) -> None:

        dataset = datasets.CIFAR10(root=data_path, 
                                         train=train, 
                                         transform=transform, download=True)
        
        if transform is None:
            transform = transforms.Compose([
                    transforms.ToPILImage(),  # Convert numpy array to PIL Image
                    transforms.Resize((224, 224)),  # Resize the image to 224x224
                    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])  # Normalize the data
                ])

        super().__init__(data=dataset.data, targets=dataset.targets, transform=transform)


if __name__ == '__main___':
    import yaml
    from Modules.wrap_model import WrapModel
    from Modules.ultis import get_trainer, seed_everything, get_class_overlay, get_metrics, get_optimizer, get_loss_function, get_lr_scheduler_config, get_bboxes_overlay
    import torchvision
    import torch
    from pytorch_lightning.loggers import NeptuneLogger


    seed_everything(44)
    torch.backends.cudnn.benchmark = True # keep True if all the input have same size.
    torch.set_float32_matmul_precision('medium')


    with open('config/new-config.yml', 'r') as stream:
        PARAMS = yaml.safe_load(stream)

    logger = NeptuneLogger(
            project=PARAMS['logger']['project'],
            # api_key=PARAMS['logger']['api_key'],
            tags=PARAMS['logger']['tags'],
            log_model_checkpoints=False)

    #load data
    # train_dataset = CIFAR10(data_path=PARAMS['dataset_settings']['path'], train=True)
    # test_dataset = CIFAR10(data_path=PARAMS['dataset_settings']['path'], train=False)

    train_dataset = PennFudanDataset(data_path=PARAMS['dataset_settings']['path'])
    test_dataset = PennFudanDataset(data_path=PARAMS['dataset_settings']['path'])

    data_loader = DataModule(train_dataset,
                            test_dataset,
                            batch_size=PARAMS['dataset_settings']['batch_size'],
                            num_workers=PARAMS['dataset_settings']['num_workers'],
                            collate_fn=collate_fn_dict)

    # classification_model = Classifier.ClassificationModel(model=torchvision.models.resnet18,
    #                                                       weight=torchvision.models.ResNet18_Weights.DEFAULT,
    #                                                       is_freeze=PARAMS['architect_settings']['is_freeze'],
    #                                                       is_full=PARAMS['architect_settings']['is_full'],
    #                                                       n_cls=PARAMS['dataset_settings']['n_cls'])

    detection_model = Detector.DetectionModel(model=torchvision.models.detection.fasterrcnn_resnet50_fpn,
                                            weight=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                            is_freeze=PARAMS['architect_settings']['is_freeze'],
                                            is_full=PARAMS['architect_settings']['is_full'],
                                            n_cls=PARAMS['dataset_settings']['n_cls'])
                                                        
    # create model
    model = WrapModel(  model=detection_model,
                        optimizer_fn=get_optimizer(PARAMS['training_settings']['optimizer'],
                                                PARAMS['training_settings']['lr']),
                        loss_fn=get_loss_function(PARAMS['training_settings']['loss']),
                        train_metrics_fn=get_metrics(PARAMS['training_settings']['metrics'],
                                            PARAMS['dataset_settings']['n_cls'],
                                            prefix='train_'),
                        val_metrics_fn=get_metrics(PARAMS['training_settings']['metrics'],
                                            PARAMS['dataset_settings']['n_cls'],
                                            prefix='val_'),
                        lr_scheduler_fn=get_lr_scheduler_config(PARAMS['training_settings']['monitor'],
                                                                PARAMS['training_settings']['lr_scheduler']),
                        log_fn=get_bboxes_overlay
                    )

    trainer = get_trainer(logger=logger,
                        gpu_ids=PARAMS['training_settings']['gpu_ids'],
                        monitor=PARAMS['training_settings']['monitor'],
                        mode=PARAMS['training_settings']['mode'],
                        max_epochs=PARAMS['training_settings']['max_epochs'],
                        ckpt_path=PARAMS['training_settings']['ckpt_path'],
                        early_stopping=PARAMS['training_settings']['early_stopping'])

    # train
    trainer.fit(model, data_loader)
    # test
    # trainer.test(model, data_loader)