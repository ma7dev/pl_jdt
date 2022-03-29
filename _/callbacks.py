import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

# class LogPredictionSamplesCallback(Callback):
    
#     def on_validation_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         """Called when the validation batch ends."""
 
#         # `outputs` comes from `LightningModule.validation_step`
#         # which corresponds to our model predictions in this case
        
#         # Let's log 20 sample image predictions from the first batch
#         if batch_idx == 0:
#             n = 20
#             x, y = batch
#             images = [img for img in x[:n]]
#             captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' 
#                 for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            
#             # Option 1: log images with `WandbLogger.log_image`
#             wandb_logger.log_image(
#                 key='sample_images', 
#                 images=images, 
#                 caption=captions)


#             # Option 2: log images and predictions as a W&B Table
#             columns = ['image', 'ground truth', 'prediction']
#             data = [[wandb.Image(x_i), y_i, y_pred] f
#                 or x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
#             wandb_logger.log_table(
#                 key='sample_table',
#                 columns=columns,
#                 data=data)