from abc import abstractmethod

import wandb

class Callback:

    def __init__(self) -> None:
        pass

    @abstractmethod
    def at_epoch_end(self, **kwargs):
        pass


class WandbCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
    
    def at_epoch_end(self, logs):
        wandb.log(logs)
        
