from pytorch_lightning.callbacks.progress import Tqdm


class CustomProgressBar(Tqdm):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics()
        items.pop("v_num", None)
        return items