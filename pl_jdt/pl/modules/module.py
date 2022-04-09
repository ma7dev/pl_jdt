# torch
import torch
import torch.nn as nn

# python lightning
import pytorch_lightning as pl
import torchmetrics

# modules
from pl_jdt.models.model import Net
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
        self.model = Net()
        self.criteria = nn.CrossEntropyLoss()
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#
        self.metric = {
            'train': torchmetrics.Accuracy(),
            'val': torchmetrics.Accuracy(),
            'test': torchmetrics.Accuracy()
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
            }
        }
        self.kwargs = kwargs

    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        # params = self.model.get_params()
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    # logging
    def logging_step(self, loss, acc, mode):
        self.log(
            f"{mode}/step/loss", 
            loss, 
            on_step=True, rank_zero_only=True
        )
        self.log(
            f"{mode}/step/acc", 
            acc, 
            on_step=True, rank_zero_only=True
        )

        # self.logger.log_metrics(
        #     {f"{mode}/step/acc": acc}, 
        #     self.global_step + 1
        # )
        # self.logger.log_metrics(
        #     {f"{mode}/step/loss": loss}, 
        #     self.global_step + 1
        # )
    def logging_epoch(self, mode):
        self.log(
            f"{mode}/epoch/loss", 
            self.history['loss'][mode][-1], 
            on_epoch=True, prog_bar=True, rank_zero_only=True
        )
        self.log(
            f"{mode}/epoch/acc", 
            self.history['acc'][mode][-1], 
            on_epoch=True, prog_bar=True, rank_zero_only=True
        )
        # self.logger.log_metrics(
        #     {f"{mode}/epoch/acc": self.history['acc'][mode][-1]}, 
        #     self.current_epoch + 1
        # )
        # self.logger.log_metrics(
        #     {f"{mode}/epoch/loss": self.history['loss'][mode][-1]}, 
        #     self.current_epoch + 1
        # )
    
    # compute
    def compute_metrics(self, outputs, labels, mode):  
        if labels.device != self.metric[mode].device:
            self.metric[mode].to(self.device)
        acc = self.metric[mode](outputs, labels)
        return acc

    # update history
    def update_history(self,outputs,mode):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = self.metric[mode].compute()
        self.history['loss'][mode].append(loss)
        self.history['acc'][mode].append(acc)
        self.metric[mode].reset()
        return loss, acc
    
    # _step
    def _step(self, batch, batch_idx, mode):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        return {'loss': loss, 'preds': y_hat.detach(), 'target': y}
    def training_step(self, batch, batch_idx): return self._step(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx): return self._step(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx): return self._step(batch, batch_idx, 'test')
    
    # _step_end
    def _step_end(self, step_outputs, mode):
        acc = self.compute_metrics(
            step_outputs['preds'], 
            step_outputs['target'], 
            mode
        )
        loss = step_outputs['loss']
        # self.logging_step(loss, acc, mode)
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
        x, y = batch
        pred = self.model(x)
        return pred
        # # log 6 example images
        # # or generated text... or whatever
        # sample_imgs = x[:6]
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image('example_images', grid, 0)

        # # calculate acc
        # labels_hat = torch.argmax(out, dim=1)
        # test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # # log the outputs!
        # self.log_dict({'test_loss': loss, 'test_acc': test_acc})
    # def test_step_end(self, output_results):
    #     # this out is now the full size of the batch
    #     all_test_step_outs = output_results.out
    #     loss = nce_loss(all_test_step_outs)
    #     self.log("test_loss", loss)
    # def test_epoch_end(self, step_outputs):
    #     # do something with the outputs of all test batches
    #     all_test_preds = step_outputs.predictions

    #     some_result = calc_all_results(all_test_preds)
    #     self.log(some_result)