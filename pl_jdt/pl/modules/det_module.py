import sys, math
# torch
import torch
import torch.nn as nn

# python lightning
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.detection.map import MeanAveragePrecision

# modules
from pl_jdt.models.faster_rcnn import get_detection_model
import pl_jdt.utils.references.detection.utils as utils
losses = {
    'cross_entropy': {
        'module': nn.CrossEntropyLoss(),
        'args': []
    },
    'mse': {
        'module':nn.MSELoss(),
        'args': []
    }
}
OPTIMIZERS = {
    'adam': {
        'module': torch.optim.Adam,
        'args': ['lr', 'betas', 'eps']
    },
    'sgd': {
        'module': torch.optim.SGD,
        'args': ['lr', 'momentum', 'weight_decay']
    },
}
LR_SCHEDULER = {
    'step': {
        'module': torch.optim.lr_scheduler.StepLR,
        'args': ['step_size', 'gamma']
    },
    'multi_step': {
        'module': torch.optim.lr_scheduler.MultiStepLR,
        'args': ['milestones', 'gamma']
    }
}

class LitModule(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._kwargs = kwargs
        self.set_model_kwargs()
        self.model = get_detection_model(**self._model_kwargs)
        self.criteria = nn.CrossEntropyLoss()
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#
        # self._map = [''] # pick mAP
        self.metric = {
            'train': MeanAveragePrecision(),
            'val': MeanAveragePrecision(),
            'test': MeanAveragePrecision()
        }

        self.reported = ['loss', 'map', 'mar_1']
        self.kwargs = kwargs
        self.init()
        self.automatic_optimization = False
    
    def set_model_kwargs(self):
        self._model_kwargs = {
            'num_classes': self._kwargs['num_classes'] if 'num_classes' in self._kwargs.keys() else 2,
            'hidden_layer': self._kwargs['hidden_layer'] if 'hidden_layer' in self._kwargs.keys() else 256,
            # 'pretrained': self._kwargs['pretrained'] if 'pretrained' in self._kwargs.keys() else True,
            # 'backbone': self._kwargs['backbone'] if 'backbone' in self._kwargs else.keys() else 'resnet50',
        }
    def init(self):
        self._optimizer = self.kwargs['optimizer'] if 'optimizer' in self.kwargs.keys() else {'name': 'sgd', 'params': {'lr': 0.00001, 'weight_decay': 0.0005,'momentum': 0.9}}
        self._lr_scheduler = self.kwargs['lr_scheduler'] if 'lr_scheduler' in self.kwargs.keys() else {'name': 'step', 'params': {'step_size': 10, 'gamma': 0.1}}
        self._warmup = self.kwargs['warmup'] if 'warmup' in self.kwargs.keys() else False
        self.len_train_dataloader = self.kwargs['len_train_data_loader'] if 'len_train_data_loader' in self.kwargs.keys() else exit(-1)
    def forward(self, images, target=None):
        if target is None:
            return self.model(images)
        return self.model(images, target)
    
    def warmup_lr_scheduler(self, optimizer):
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, self.len_train_dataloader - 1)
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = OPTIMIZERS[self._optimizer['name']]['module'](params, **self._optimizer['params'])
        lr_scheduler = LR_SCHEDULER[self._lr_scheduler['name']]['module'](optimizer, **self._lr_scheduler['params'])
        lr_scheduler_warmup = self.warmup_lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler, lr_scheduler_warmup]
    
    # logging
    def logging_step(self, loss, acc, loss_dict, mode):
        if loss != None:
                self.log(
                    f"{mode}/step/loss", 
                    loss, 
                    on_step=True, rank_zero_only=True
                )
        if acc != None:
            self.log(
                f"{mode}/step/acc", 
                acc, 
                on_step=True, rank_zero_only=True
            )
        if loss_dict != {}:
            for k, v in loss_dict.items():
                self.log(
                    f"{mode}/step/{k}", 
                    v, 
                    on_step=True, rank_zero_only=True
                )
    
    def logging_epoch(self, outputs, mode):
        if mode == 'train':
            loss = torch.stack([x['loss'] for x in outputs]).mean()
            self.log(
                f"{mode}/epoch/loss", 
                loss, 
                on_epoch=True, prog_bar=True, rank_zero_only=True
            )
            if 'loss_dict' in outputs[-1]:
                for k in outputs[-1]['loss_dict'].keys():
                    prog_bar = True if k in self.reported else False
                    loss = torch.stack([x['loss_dict'][k] for x in outputs]).mean()
                    self.log(
                        f"{mode}/epoch/{k}", 
                        loss, 
                        on_epoch=True, prog_bar=prog_bar, rank_zero_only=True
                    )
        else:
            acc = self.metric[mode].compute()
            if isinstance(acc, dict):
                for k, v in acc.items():
                    prog_bar = True if k in self.reported else False
                    self.log(
                        f"{mode}/epoch/{k}", 
                        v, 
                        on_epoch=True, prog_bar=prog_bar, rank_zero_only=True
                    )
            else:
                self.log(
                    f"{mode}/epoch/acc", 
                    acc, 
                    on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=sync_dist
                )
            self.metric[mode].reset()
    
    # compute
    def compute_metrics(self, outputs, targets, mode):  
        if targets[0]['boxes'].device != self.metric[mode].device:
            self.metric[mode].to(self.device)
        acc = self.metric[mode](outputs, targets)
        return acc
    
    # _step
    def _step(self, batch, batch_idx, mode):
        if mode == 'train':
            optimizer = self.optimizers()
            
            images, tagets = batch
            loss_dict = self.forward(images, tagets)
            loss = sum(loss for loss in loss_dict.values())
            loss_dict = {k: v.detach() for k, v in loss_dict.items()}

            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

            if self._warmup and self.current_epoch == 0: 
                _, lr_scheduler_warmup = self.lr_schedulers()
                lr_scheduler_warmup.step()
            return {'loss': loss, 'loss_dict': loss_dict}
        else:
            images, targets = batch
            outputs = self.forward(images)
            return {'outputs': outputs, 'targets': targets}
    def training_step(self, batch, batch_idx): return self._step(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx): return self._step(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx): return self._step(batch, batch_idx, 'test')
    
    # _step_end
    def _step_end(self, step_outputs, mode):
        loss, acc, loss_dict = None, None, {}
        if mode == 'train':
            loss = step_outputs['loss']
            if 'loss_dict' in step_outputs:
                loss_dict = step_outputs['loss_dict']
        else:
            acc = self.compute_metrics(
                step_outputs['outputs'], 
                step_outputs['targets'], 
                mode
            )
        self.logging_step(loss, acc, loss_dict, mode)
    def training_step_end(self, step_outputs): self._step_end(step_outputs, 'train')
    def validation_step_end(self, step_outputs): self._step_end(step_outputs, 'val')
    def test_step_end(self, step_outputs): self._step_end(step_outputs, 'test')
    
    # _epoch_end
    def _epoch_end(self, outputs, mode):
        if mode == 'train':
            lr_scheduler, _ = self.lr_schedulers()
            lr_scheduler.step()
        self.logging_epoch(outputs, mode)
    def training_epoch_end(self, outputs): self._epoch_end(outputs, 'train')
    def validation_epoch_end(self, outputs): self._epoch_end(outputs, 'val')
    def test_epoch_end(self, outputs): self._epoch_end(outputs, 'test')

    def predict_step(self, batch, batch_idx):
        images = batch
        pred = self.model(images)
        return pred