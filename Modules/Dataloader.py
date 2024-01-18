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
    def __init__(self, dataset,
                 data_settings, 
                 training_settings):
        super().__init__()

        

        if(self.dataset in ["Apple", "PennFudan"]):
            self.collate_fn = collate_fn

       
    def _split(self, data, size=[0.8, 0.1, 0.1], random_state=42):
        '''
        Split data into train and test set
        Args:
            data: data to be split
            test_size: size of test set
            shuffle: whether to shuffle data before splitting
            random_state: random seed
        Returns:
            train_data: training set
            test_data: test set
        '''
        generator = torch.Generator().manual_seed(random_state)
        train_data, val_data, test_data = torch.utils.data.random_split(data, size, generator=generator)
        return train_data, val_data, test_data
        
    def setup(self, stage: str):

        if stage == "fit":
            self.Train_dataset = self.data_class[self.dataset](mode="train", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.train_transform)
            self.Val_dataset = self.data_class[self.dataset](mode="val", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.val_transform)
                
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.Test_dataset = self.data_class[self.dataset](mode="test", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.val_transform)
           
    def train_dataloader(self):
        return DataLoader(self.Train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=8, 
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.Val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=8, 
                          collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.Test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=8, 
                          collate_fn=self.collate_fn)
