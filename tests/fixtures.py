import os
import pytest

import hydra
from omegaconf import DictConfig, OmegaConf

# torch
import torch
from torch.utils import data
import torchvision
from torchvision import transforms

import pytorch_lightning as pl

@pytest.fixture(scope="session")
def common():
    return {
        'project_name': 'pl_jdt',
        'output_path': './output',
        'data_path': './data',
        'exp_name': 'pytest',
    }
