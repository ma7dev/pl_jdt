import os, sys
import yaml

from .fixtures import *

@pytest.fixture
def set_config():
    os.environ['HYDRA_FULL_ERROR'] = '1'

@pytest.fixture
def set_seed():
    seed = 42
    pl.seed_everything(seed, workers=True)