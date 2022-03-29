import random, string

import wandb
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only


class MyLogger(LightningLoggerBase):
    def __init__(self,**kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.init()
    
    def init(self):
        self.name = self.kwargs['name'] if 'name' in self.kwargs.keys() else 'default'
        self.name = f"{self.name}_{''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}"
        self.save_dir = self.kwargs['save_dir'] if 'save_dir' in self.kwargs.keys() else './output'
        self.project = self.kwargs['project'] if 'project' in self.kwargs.keys() else 'test'


    @property
    def name(self):
        return self.name

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass