import os, sys, yaml
import hydra
from omegaconf import DictConfig, OmegaConf

# torch
from torchvision import transforms

# others
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, DeviceStatsMonitor

# modules
from pl_jdt.pl.modules.module import LitModule
from pl_jdt.pl.datasets.dataset import LitDataset
import pl_jdt.utils.utils as utils

# rich
from rich import pretty, traceback
pretty.install()
traceback.install(suppress=[
    hydra, 
    pl,
])

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

def init(cfg, common={}):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    common['project_name'] = cfg.project.name
    common['output_path'] = f"{cfg.project.root_dir}/{cfg.project.output_dir}"
    common['data_path'] = f"{cfg.project.root_dir}/{cfg.project.data_dir}"
    common['exp_name'] = cfg.exp.name
    # exp_path = '{save_dir}/{today}/{curr_time}-{exp_name}-{random_str}'
    # example: ./output/2020-04-24/12-14-test-L1GV1
    common['exp_path'], common['run_name'] = utils.exp(common['output_path'], common['exp_name'])
    # set seed
    seed = cfg.exp.seed if 'seed' in cfg.exp.keys() else 42
    pl.seed_everything(seed, workers=True)
    return common

def get_dataset_kwargs(cfg):
    return {
        'batch_size': cfg.exp.train.batch_size if 'batch_size' in cfg.exp.train.keys() else 32,
        'val_batch_size': cfg.exp.val.batch_size if 'val_batch_size' in cfg.exp.val.keys() else cfg.exp.train.batch_size,
        'num_workers': cfg.exp.num_workers if 'num_workers' in cfg.exp.keys() else 4,
    }

def get_dataset(dataset_kwargs, common):
    dataset = LitDataset(common['data_path'], get_transforms, **dataset_kwargs)
    dataset.prepare_data()
    dataset.setup()
    return dataset

def get_module_kwargs(cfg):
    module_kwargs = {}
    checkpoint_path = cfg.exp.checkpoint_path if 'checkpoint_path' in cfg.exp.keys() else False
    if checkpoint_path:
        module_kwargs.update({
            'checkpoint_path': checkpoint_path,
        })
    else:
        module_kwargs.update({
            'optim': cfg.exp.train.optim if 'optim' in cfg.exp.train.keys() else 'adam',
        })
    return module_kwargs

def get_module(module_kwargs):
    if 'checkpoint_path' in module_kwargs.keys():
        return LitModule.load_from_checkpoint(**module_kwargs)
    return LitModule(**module_kwargs)

def get_trainer_kwargs(cfg):
    trainer_kwargs = {
        'max_epochs': cfg.exp.max_epochs if 'max_epochs' in cfg.exp.keys() else 10,
        # int: n = check validation set every 1000 training batches
        # float: n = check validation set 1/n times during a training epoch
        'val_check_interval': cfg.exp.val_check_interval if 'test' in cfg.exp.keys() and cfg.exp.test and 'val_check_interval' in cfg.exp.keys() else 1.0,
        # int: n = run val every n epoch
        'check_val_every_n_epoch': cfg.exp.check_val_every_n_epoch if 'test' in cfg.exp.keys() and cfg.exp.test and 'check_val_every_n_epoch' in cfg.exp.keys() else 1,
        # int: n = log every n steps
        'log_every_n_steps': cfg.exp.log_every_n_steps if 'log_every_n_steps' in cfg.exp.keys() else 100,
        # bool: True = run 1 train, val, test batch
        # int: n = run n train, val, test batches
        'fast_dev_run': cfg.exp.fast_dev_run if 'fast_dev_run' in cfg.exp.keys() else False,
        # int: n = check runs n batches of val before training
        'num_sanity_val_steps': cfg.exp.num_sanity_val_steps if 'num_sanity_val_steps' in cfg.exp.keys() else 2,
        # simple, advanced - https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html
        'profiler': cfg.exp.profiler if 'profiler' in cfg.exp.keys() else None,
        # float: n = n/100 of the training data is trained on every epoch
        # int: n = n of the training data is trained on every epoch
        'overfit_batches': cfg.exp.overfit_batches if 'overfit_batches' in cfg.exp.keys() else 0,
        # 'default_root_dir': f'{exp_path}/ckpts',
        # 'max_steps': cfg.exp.train.max_steps if 'max_step' in cfg.exp.train else 10,
        # 'accumulate_grad_batches': cfg.exp.accumulate_grad_batches if 'accumulate_grad_batches' in cfg.exp else 1,
        # 'auto_lr_find': cfg.exp.auto_lr_find if 'auto_lr_find' in cfg.exp else False,
        # 'benchmark': cfg.exp.benchmark if 'benchmark' in cfg.exp else False,
        # 'deterministic': cfg.exp.deterministic if 'deterministic' in cfg.exp else False,
    }
    # trainer_kwargs.update({
    #     'devices': 1,
    #     'accelerator': 'gpu'
    # })
    # if 'devices' in cfg.exp.keys() and len(cfg.exp.devices) > 1:
    #         trainer_kwargs.update({
    #             'devices': len(cfg.exp.devices),
    #             'accelerator': cfg.exp.accelerator if 'accelerator' in cfg.exp.keys() else 'gpu',
    #             'strategy': cfg.exp.strategy if 'strategy' in cfg.exp else 'ddp',
    #             'amp_backend': cfg.exp.amp_backend if 'amp_backend' in cfg.exp.keys() else 'native',
    #             'sync_batchnorm': cfg.exp.sync_batchnorm if 'sync_batchnorm' in cfg.exp.keys() else False,
    #         })
    # else: 
    #     trainer_kwargs.update({
    #         'devices': cfg.exp.devices if 'devices' in cfg.exp.keys() else [0],
    #         'accelerator': cfg.exp.accelerator if 'accelerator' in cfg.exp.keys() else 'gpu',
    #     })
    return trainer_kwargs

def get_loggers(cfg, common):
    loggers = {}
    enabled = {}
    if 'csv' in cfg.exp.keys() and cfg.exp.csv:
        csv_kwargs = {
            'save_dir': f"{common['output_path']}/csv_logs",
            'name': common['run_name']
        }
        enabled['csv'] = csv_kwargs
        loggers['csv'] = CSVLogger(**csv_kwargs)
    if 'tb' in cfg.exp.keys() and cfg.exp.tb:
        tb_kwargs = {
            'save_dir': f"{common['output_path']}/tb_logs",
            'name': common['run_name']
        }
        enabled['tb'] = tb_kwargs
        loggers['tb'] = TensorBoardLogger(**tb_kwargs)
    if 'wandb' in cfg.exp.keys() and cfg.exp.wandb:
        wandb_kwargs = {
            'project': common['project_name'],
            'name': common['run_name'],
            'save_dir': f"{common['output_path']}"
        }
        enabled['wandb'] = wandb_kwargs
        loggers['wandb'] = WandbLogger(**wandb_kwargs)
    print(f'Enabled loggers:\n{OmegaConf.to_yaml(enabled)}\n\n')
    return loggers

def get_callbacks(cfg, common):
    callbacks = {}
    enabled = {}
    if 'checkpoints' in cfg.exp.keys():
        checkpoints_kwargs = {
            'dirpath': cfg.exp.checkpoints.dirpath if 'dirpath' in cfg.exp.checkpoints.keys() else f"{common['exp_path']}/ckpts",
            # 'every_n_epochs': cfg.exp.checkpoints.every_n_epochs if 'every_n_epochs' in cfg.exp.checkpoints.keys() else 1,
            'monitor': cfg.exp.checkpoints.monitor if 'monitor' in cfg.exp.checkpoints.keys() else 'val/epoch/loss',
            # 'filename': 'sample-mnist-{epoch:03d}-{val_loss:.2f}',
        }
        enabled['checkpoints'] = checkpoints_kwargs
        callbacks['checkpoints'] = ModelCheckpoint(**checkpoints_kwargs)
    else:
        checkpoints_kwargs = {
            'dirpath': f"{common['exp_path']}/ckpts",
            'monitor': 'val/epoch/loss',
        }
        enabled['checkpoints'] = checkpoints_kwargs
        callbacks['checkpoints'] = ModelCheckpoint(**checkpoints_kwargs)
    if 'device_stats_monitor' in cfg.exp.keys() and cfg.exp.device_stats_monitor:
        enabled['device_stats_monitor'] = True
        callbacks['device_stats_monitor'] = DeviceStatsMonitor()
    if 'early_stopping' in cfg.exp.keys():
        early_stopping_kwargs = {
            'monitor': cfg.exp.early_stopping.monitor if 'monitor' in cfg.exp.early_stopping.keys() else "val/epoch/loss",
            'patience': cfg.exp.early_stopping.patience if 'patience' in cfg.exp.early_stopping.keys() else 3,
        }
        enabled['early_stopping'] = early_stopping_kwargs
        callbacks['early_stopping'] = EarlyStopping(**early_stopping_kwargs)
    if 'model_summary' in cfg.exp.keys():
        model_summary_kwargs = {
            'max_depth': cfg.exp.model_summary.max_depth if 'max_depth' in cfg.exp.model_summary.keys() else 1,
        }
        enabled['model_summary'] = model_summary_kwargs
        callbacks['model_summary'] = ModelSummary(**model_summary_kwargs)
    print(f'Enabled callbacks:\n{OmegaConf.to_yaml(enabled)}\n\n')
    return callbacks

@hydra.main(config_path="../cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    # init
    common = init(cfg)

    # Dataset
    dataset_kwargs = get_dataset_kwargs(cfg)
    dataset = get_dataset(dataset_kwargs, common)

    # Module
    module_kwargs = get_module_kwargs(cfg)
    module = get_module(module_kwargs)

    # Trainer
    trainer_kwargs = get_trainer_kwargs(cfg)

    # Loggers
    loggers = get_loggers(cfg, common)
    callbacks = get_callbacks(cfg, common)

    trainer_kwargs['callbacks'] = [callback for callback in callbacks.keys()]
    trainer_kwargs['logger'] = [logger for logger in loggers.keys()]
    # print(OmegaConf.to_yaml(trainer_kwargs))
    trainer_kwargs['callbacks'] = [callback for callback in callbacks.values()]
    trainer_kwargs['logger'] = [logger for logger in loggers.values()]

    # exit()
    trainer = pl.Trainer(**trainer_kwargs)

    # fit
    trainer.fit(module, dataset.train_dataloader(), dataset.val_dataloader())
if __name__ == "__main__":
    main()