import os, sys, yaml

# basic
import matplotlib.pyplot as plt
from PIL import Image

# torch
import torch
# modules
ROOT_DIR = None
with open('cfg/project/default.yaml', 'r') as f: ROOT_DIR = yaml.load(f, Loader=yaml.FullLoader)['root_dir']
sys.path.insert(0, os.path.abspath(f"{ROOT_DIR}"))

from src.data.datasets.penn_fundan import PennFudanDataset
from src.utils.references.detection.engine import train_one_epoch, evaluate
import src.utils.references.detection.utils as utils
import src.utils.references.detection.transforms as T
# import src.utils.transforms as T
from src.models.mask_rcnn import MaskRCNN

# global
DATA_PATH = f'{ROOT_DIR}/data/PennFudanPed'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# use our dataset and defined transformations
dataset = PennFudanDataset(DATA_PATH, get_transform(train=True))
dataset_test = PennFudanDataset(DATA_PATH, get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

model_kwargs = {
    'num_classes': 2,
    'hidden_layer': 256
}
model = MaskRCNN(**model_kwargs)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

num_epochs = 10
print_freq = 10
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)