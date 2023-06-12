from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
from torch import nn
from torchvision.transforms._presets import SemanticSegmentation, ObjectDetection
from functools import partial

import numpy as np
import os, glob
import PIL.Image as Image


def normalize_image(image):
    xmin = np.min(image)
    xmax = np.max(image)
    return (image - xmin)/ (xmax - xmin)

def collate_fn(batch):
    return tuple(zip(*batch))

class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """
    def __call__(self, img):
        return (img - img.mean(dim=(1,2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1,2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class LungCTscan(Dataset):
    def __init__(self, data_dir, transform=None, imgsize=224):
        self.img_list = sorted(glob.glob(data_dir + '/2d_images/*.tif'))
        self.mask_list = sorted(glob.glob(data_dir + '/2d_masks/*.tif'))
        self.transform = transform
        if(self.transform is None):
            self.transformImg = partial(SemanticSegmentation, resize_size=imgsize)()
            self.transformAnn = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                                    transforms.ToTensor()])
        
    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        # load image
        image = Image.open(image_path).convert('RGB')
        # resize image with 1 channel

        # load image
        mask = Image.open(mask_path).convert('L')

        if self.transform is None:
            image = self.transformImg(image)
            mask = self.transformAnn(mask)
        else:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask.squeeze(0).long()


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, imgsize=224):
        self.root = root
        self.transform = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        if(self.transform is None):
            self.transform = partial(ObjectDetection)()

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
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

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
        
class CIFAR10read(Dataset):
    """Customized dataset loader"""
    def __init__(self, mode, data_path, transform=None, imgsize=224):

        if(mode == 'test'):
            dataset = CIFAR10(root=data_path, download=True, train=False)
        else:
            dataset = CIFAR10(root=data_path, download=True, train=True)
        data = getattr(dataset, 'data')
        labels = getattr(dataset, 'targets')
        if(mode == 'train'):
            data = data[:40000]
            labels = labels[:40000]
        elif(mode == 'val'):
            data = data[40000:50000]
            labels = labels[40000:50000]
        self.transform = transform
        self.input_images = np.array(data, np.uint8)
        self.input_labels = np.array(labels)
        if(self.transform == None):
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
      
        images = self.transform(images)
        return images, labels

    
#---------------------------------------lOSS FUNCTION-----------------------------------------------

def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


class DiceLoss(nn.Module):

    def __init__(self, multiclass = False):
        super(DiceLoss, self).__init__()
        if(multiclass):
            self.dice_loss = multiclass_dice_coeff
        else:
            self.dice_loss = dice_coeff

        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, seg, target):
        bce = self.BCE(seg, target)
        seg = torch.sigmoid(seg)
        dice = 1 - self.dice_loss(seg, target)
        # Dice loss (objective to minimize) between 0 and 1
        return dice + bce
       