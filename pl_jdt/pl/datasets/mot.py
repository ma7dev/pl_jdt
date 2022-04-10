from typing import Optional
import os

# torch
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split

# pl
import pytorch_lightning as pl

# module
from pl_jdt.data.datasets.mot import MOTObjDetect
import pl_jdt.utils.references.detection.transforms as T
import pl_jdt.utils.references.detection.utils as utils

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
        self.num_workers = self.kwargs['num_workers'] if 'num_workers' in self.kwargs.keys() else 4; print(f"num_workers: {self.num_workers}")
        self.batch_size = self.kwargs['batch_size'] if 'batch_size' in self.kwargs.keys() else 2; print(f"batch_size: {self.batch_size}")
        self.val_batch_size = self.kwargs['val_batch_size'] if 'val_batch_size' in self.kwargs.keys() else 2; print(f"val_batch_size: {self.val_batch_size}")
        self.train_split = self.kwargs['train_split'] if 'train_split' in self.kwargs.keys() else None; print(f"train_split: {self.train_split}")
        self.val_split = self.kwargs['val_split'] if 'val_split' in self.kwargs.keys() else None; print(f"val_split: {self.val_split}")
        self.test_split = self.kwargs['test_split'] if 'test_split' in self.kwargs.keys() else None; print(f"test_split: {self.test_split}")
    def prepare_data(self):
        # if not os.path.isdir(self.data_dir): os.makedirs(self.data_dir)
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MOTObjDetect(f"{self.data_dir}/train", self.get_transforms('train'),split_seqs=self.train_split)
            self.val_dataset = MOTObjDetect(f"{self.data_dir}/train", self.get_transforms('val'),split_seqs=self.val_split)
        if stage == "test" or stage is None:
            self.test_dataset = MOTObjDetect(f"{self.data_dir}/train", self.get_transforms('test'),split_seqs=self.test_split)
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
