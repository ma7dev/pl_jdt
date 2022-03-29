from typing import Optional
import os

# torch
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split

# pl
import pytorch_lightning as pl

# module
from src.data.datasets.penn_fundan import PennFudanDataset
import src.utils.references.detection.transforms as T
import src.utils.references.detection.utils as utils

class LitDataset(pl.LightningModule):
    def __init__(self, data_dir: str = "./data", get_transforms = None, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.get_transforms = get_transforms
        self.kwargs = kwargs
        self.init()
    def init(self):
        if self.get_transforms is None:
            def get_transforms(mode):
                transforms = []
                transforms.append(T.ToTensor())
                if mode == 'train':
                    transforms.append(T.RandomHorizontalFlip(0.5))
                return T.Compose(transforms)
            self.get_transforms = get_transforms
        self.num_workers = self.kwargs['num_workers'] if 'num_workers' in self.kwargs.keys() else 4
        self.batch_size = self.kwargs['batch_size'] if 'batch_size' in self.kwargs.keys() else 2
        self.val_batch_size = self.kwargs['val_batch_size'] if 'val_batch_size' in self.kwargs.keys() else 1
        if not os.path.isdir(self.data_dir): os.makedirs(self.data_dir)
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_dataset = PennFudanDataset(self.data_dir, self.get_transforms('train'))
            val_dataset = PennFudanDataset(self.data_dir, self.get_transforms('val'))
            # split the dataset in train and test set
            indices = torch.randperm(len(train_dataset)).tolist()
            self.train_dataset = torch.utils.data.Subset(train_dataset, indices[:-50])
            self.val_dataset = torch.utils.data.Subset(val_dataset, indices[-50:])
        if stage == "test" or stage is None:
            self.test_dataset = PennFudanDataset(self.data_dir, self.get_transforms('test'))
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=utils.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.val_batch_size, num_workers=self.num_workers, collate_fn=utils.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.val_batch_size, num_workers=self.num_workers, collate_fn=utils.collate_fn)

    # def teardown(self):
    #     # clean up after fit or test
    #     # called on every process in DDP
    #     pass
