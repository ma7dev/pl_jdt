# https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#plugins
from pytorch_lightning.plugins.environments import ClusterEnvironment


class MyCluster(ClusterEnvironment):
    def main_address(self):
        return your_main_address

    def main_port(self):
        return your_main_port

    def world_size(self):
        return the_world_size