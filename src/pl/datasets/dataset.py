from typing import Optional
import os

# torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split

# pl
import pytorch_lightning as pl

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
                if mode == 'train':
                    return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                else:
                    return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
            self.get_transforms = get_transforms
        self.num_workers = self.kwargs['num_workers'] if 'num_workers' in self.kwargs.keys() else 4
        self.batch_size = self.kwargs['batch_size'] if 'batch_size' in self.kwargs.keys() else 32
        self.val_batch_size = self.kwargs['val_batch_size'] if 'val_batch_size' in self.kwargs.keys() else self.batch_size
        if not os.path.isdir(self.data_dir): os.makedirs(self.data_dir)
    
    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.MNIST(self.data_dir, train=True,download=True, transform=self.get_transforms('train'))
            self.val_dataset = datasets.MNIST(self.data_dir, train=False,download=True, transform=self.get_transforms('val'))
        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.get_transforms('test'))
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers)

    # def teardown(self):
    #     # clean up after fit or test
    #     # called on every process in DDP
    #     pass
