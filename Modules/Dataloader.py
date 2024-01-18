from torch.utils.data import Dataset
# from torchvision import transforms
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np


def normalize_image(image):
    xmin = np.min(image)
    xmax = np.max(image)
    return (image - xmin)/(xmax - xmin + 10e-6)

def collate_fn(batch):
    return tuple(zip(*batch))

def collate_fn_dict(batch):
    # batch: list of dict
    # return: dict of list
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [sample[key] for sample in batch]
    return batch_dict
    
class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """
    def __call__(self, img):
        return (img - img.mean(dim=(1,2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1,2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class base_dataset(Dataset):
    '''
    Base dataset class for all datasets
    '''
    def __init__(self,
                data: list, 
                targets: list=None,
                transform=None) -> None:
        
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __transform__(self, sample: dict) -> dict:
        if self.transform:
            transfromed = self.transform(sample['data'])
            sample.update({'data': transfromed})

        return sample
    
    def __load_sample__(self, index) -> dict:
        
        return {'data': self.data[index],
                'target': self.targets[index],
                'original': self.data[index]}

    def __getitem__(self, index) -> dict:

        sample = self.__load_sample__(index)
        sample = self.__transform__(sample)

        return sample
      

class DataModule(LightningDataModule):
    '''
    Data Module for Train/Val/Test data loadding
    Args: 
        data_settings, training_settings: hyperparameter settings
        transform: data augmentation
    Returns:
        Train/Test/Val data loader
    '''
    def __init__(self, 
                dataset: base_dataset,
                batch_size: int,
                num_workers: int, 
                collate_fn=None):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        super().__init__()

   
    def setup(self, stage: str):
        
        if stage == "fit":
            generator = torch.Generator().manual_seed(42)
            self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, 
                                                                        lengths=[0.8, 0.2], 
                                                                        generator=generator)
            
        elif stage == "test":
            self.test_set = self.dataset
           
    def train_dataloader(self):
        return DataLoader(self.train_set, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)
