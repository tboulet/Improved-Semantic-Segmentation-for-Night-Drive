from abc import abstractmethod
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb

from improved_nightdrive.pipeline.preprocess import RandomCrop, Resize


class Callback:
    """Base class for all callbacks

    Methods:
        __call__: Applies callback
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def at_epoch_end(self, **kwargs):
        pass


class WandbCallback(Callback):
    """Wandb logger"""
    def __init__(self) -> None:
        super().__init__()
    
    def at_epoch_end(self, logs, **kwargs):
        wandb.log(logs)
        

class InferOnImage(Callback):
    """Infers prediction on a list of images"""
    def __init__(
        self,
        img_paths: List[str],
        label_paths: List[str],
        size:int = 224,
        intermediate_size: Tuple[int] = (240, 320),
        num_classes: int = 19,
        use_wandb: bool = False,
        save_path: str = "./"
    ) -> None:

        super().__init__()
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.use_wandb = use_wandb
        self.save_path = save_path

        self.imgs = []
        for i, (img, label) in enumerate(zip(self.img_paths, self.label_paths)):

            x = np.expand_dims(np.array(Image.open(img), dtype=np.float32), axis=0)
            y = np.expand_dims(np.expand_dims(np.array(Image.open(label), dtype=np.float32), axis=0), axis=-1)
            x, y = Resize(intermediate_size)(x, y)
            x, y = RandomCrop(size)(x, y)
            self.imgs.append(x)

            plt.imshow(y[0], vmin=0, vmax=num_classes-1)
            plt.savefig(os.path.join(self.save_path, f"seg_{i}.jpg"))
            plt.close()


    def at_epoch_end(self, epoch, model, **kwargs):
        
        for i, img in enumerate(self.imgs):

            ypred = model(img)
            ypred = np.argmax(ypred, axis=-1)[0].astype(np.uint8)

            plt.imshow(ypred, vmin=0, vmax=18)
            plt.savefig(os.path.join(self.save_path, f"{i}_at_epoch_{epoch}.jpg"))
            plt.close()
