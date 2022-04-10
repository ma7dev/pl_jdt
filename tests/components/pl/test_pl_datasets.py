import pytest

from torchvision import transforms
from pl_jdt.pl.datasets.dataset import LitDataset


def get_transforms(mode: str):
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

@pytest.fixture(scope="session")
def dataset_kwargs():
    return {
        'batch_size': 32,
        'val_batch_size': 1,
        'num_workers': 4
    }

@pytest.fixture(scope="session")
def dataset(dataset_kwargs, common):
    dataset = LitDataset(common['data_path'], get_transforms, **dataset_kwargs)
    dataset.prepare_data()
    dataset.setup()
    return dataset

def test_dataset(dataset, dataset_kwargs, common):
    assert common['data_path'] == dataset.data_dir
    assert get_transforms == dataset.get_transforms
    assert dataset_kwargs == dataset.kwargs