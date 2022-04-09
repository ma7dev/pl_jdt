import sys, math
# torch
import torch
import torch.nn as nn

# python lightning
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.detection.map import MeanAveragePrecision

# modules
from pl_jdt.models.mask_rcnn import get_mask_rcnn
import pl_jdt.utils.references.detection.utils as utils
# losses = {
#     'cross_entropy': {
#         'module': nn.CrossEntropyLoss(),
#         'args': []
#     },
#     'mse': {
#         'module':nn.MSELoss(),
#         'args': []
#     }
# }
# optimizers = {
#     'adam': {
#         'module': torch.optim.Adam,
#         'args': ['lr', 'betas', 'eps']
#     },
#     'sgd': {
#         'module': torch.optim.SGD,
#         'args': ['lr', 'momentum', 'weight_decay']
#     },
# }
# lr_scheduler = {
#     'step': {
#         'module': torch.optim.lr_scheduler.StepLR,
#         'args': ['step_size', 'gamma']
#     },
#     'multi_step': {
#         'module': torch.optim.lr_scheduler.MultiStepLR,
#         'args': ['milestones', 'gamma']
#     }
# }

class LitModule(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._kwargs = kwargs
        self.set_model_kwargs()
        self.model = get_mask_rcnn(**self._model_kwargs)
        self.criteria = nn.CrossEntropyLoss()
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#
        # self._map = [''] # pick mAP
        self.metric = {
            'train': MeanAveragePrecision(),
            'val': MeanAveragePrecision(),
            'test': MeanAveragePrecision()
        }
        self.history = {
            'acc': {
                'train': [],
                'val': [],
                'test': []
            },
            'loss': {    
                'train': [],
                'val': [],
                'test': []
            },
            'loss_dict': {
                'train': [],
                'val': [],
                'test': []
            },
        }
        self.reported = ['loss', 'map', 'mar_1']
        self.kwargs = kwargs
    def set_model_kwargs(self):
        self._model_kwargs = {
            'num_classes': self._kwargs['num_classes'] if 'num_classes' in self._kwargs.keys() else 2,
            'hidden_layer': self._kwargs['hidden_layer'] if 'hidden_layer' in self._kwargs.keys() else 256,
            # 'pretrained': self._kwargs['pretrained'] if 'pretrained' in self._kwargs else True,
            # 'backbone': self._kwargs['backbone'] if 'backbone' in self._kwargs else 'resnet50',

        }
    def forward(self, images, target=None):
        if target is None:
            return self.model(images)
        return self.model(images, target)
    
    def configure_optimizers(self):
        # params = self.model.get_params()
        # params = self.model.parameters()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [lr_scheduler]
    
    # logging
    def logging_step(self, loss, acc, loss_dict, mode):
        if loss is not None:
                self.log(
                    f"{mode}/step/loss", 
                    loss, 
                    on_step=True, rank_zero_only=True
                )
        if acc is not None:
            self.log(
                f"{mode}/step/acc", 
                acc, 
                on_step=True, rank_zero_only=True
            )
        if loss_dict is not {}:
            for k, v in loss_dict.items():
                self.log(
                    f"{mode}/step/{k}", 
                    v, 
                    on_step=True, rank_zero_only=True
                )

    def logging_epoch(self, mode):
        if self.history['loss'][mode] != []:
            assert self.history['loss'][mode][-1] is not None
            loss = self.history['loss'][mode][-1]
            self.log(
                f"{mode}/epoch/loss", 
                loss, 
                on_epoch=True, prog_bar=True, rank_zero_only=True
            )
        if self.history['acc'][mode] != []:
            assert self.history['acc'][mode][-1] is not None
            acc = self.history['acc'][mode][-1]
            if isinstance(acc, dict):
                for k, v in acc.items():
                    if k in self.reported:
                        self.log(
                            f"{mode}/epoch/{k}", 
                            v, 
                            on_epoch=True, prog_bar=True, rank_zero_only=True
                        )
                    else:
                        self.log(
                            f"{mode}/epoch/{k}", 
                            v, 
                            on_epoch=True, rank_zero_only=True
                        )
            else:
                self.log(
                    f"{mode}/epoch/acc", 
                    acc, 
                    on_epoch=True, prog_bar=True, rank_zero_only=True
                )
        if self.history['loss_dict'][mode] != []:
            loss_dict = self.history['loss_dict'][mode][-1]
            for k, v in loss_dict.items():
                if k in self.reported:
                    self.log(
                        f"{mode}/epoch/{k}", 
                        v, 
                        on_epoch=True, prog_bar=True, rank_zero_only=True
                    )
                else:
                    self.log(
                        f"{mode}/epoch/{k}", 
                        v, 
                        on_epoch=True, rank_zero_only=True
                    )
    
    # compute
    def compute_metrics(self, outputs, targets, mode):  
        if targets[0]['boxes'].device != self.metric[mode].device:
            self.metric[mode].to(self.device)
        acc = self.metric[mode](outputs, targets)
        return acc

    # update history
    def update_history(self, outputs, mode):
        loss, acc, loss_dict = None, None, {}
        if mode == 'train':
            assert outputs[-1]['loss'] != None
            if 'loss_dict' in outputs[-1]:
                # import pdb; pdb.set_trace()
                for k in outputs[-1]['loss_dict'].keys():
                    loss_dict[k] = torch.stack([x['loss_dict'][k] for x in outputs]).mean()
            loss = torch.stack([x['loss'] for x in outputs]).mean()
        else:
            acc = self.metric[mode].compute()
        if loss != None: 
            self.history['loss'][mode].append(loss)
        if loss_dict != {}:
            self.history['loss_dict'][mode].append(loss_dict)
        if acc != None: 
            self.history['acc'][mode].append(acc)
            self.metric[mode].reset()
    
    # _step
    def _step(self, batch, batch_idx, mode):
        if mode == 'train':
            images, tagets = batch
            loss_dict = self.forward(images, tagets)
            loss = sum(loss for loss in loss_dict.values())
            loss_dict = {k: v.detach() for k, v in loss_dict.items()}
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
        if self.local_rank != 0: return
        self.update_history(outputs, mode)
        self.logging_epoch(mode)
    def training_epoch_end(self, outputs): self._epoch_end(outputs, 'train')
    def validation_epoch_end(self, outputs): self._epoch_end(outputs, 'val')
    def test_epoch_end(self, outputs): self._epoch_end(outputs, 'test')

    def predict_step(self, batch, batch_idx):
        images = batch
        pred = self.model(images)
        return pred