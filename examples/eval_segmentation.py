from absl import app, flags
import os
import sys

sys.path.insert(1, os.path.abspath(os.path.curdir))

import matplotlib.pyplot as plt
import numpy as np
import yaml

from improved_nightdrive.pipeline.metric import ClassMeanIOU, MeanIOU
from improved_nightdrive.pipeline.pipeline import Evaluation
from improved_nightdrive.pipeline.preprocess import (
    Resize,
    ReClass,
)
from improved_nightdrive.segmentation.models import make_model

flags.DEFINE_string(
    "model_name",
    "unetmobilenetv2",
    "Available models : [unetmobilenetv2, deeplabv3, unetmobilenetv2_big]",
)
flags.DEFINE_integer("image_size", 224, "Size of the square input for cropping")
flags.DEFINE_multi_integer("intermediate_size", (225, 400), "Size before cropping")
flags.DEFINE_integer("batch_size", 10, "Size of the batches")
flags.DEFINE_string("dataset_x_path", "./BDD100K/bdd100k/day/images/", "Dataset images")
flags.DEFINE_string("dataset_y_path", "./BDD100K/bdd100k/day/labels/", "Dataset images")
flags.DEFINE_string(
    "model_weights",
    "./results/unet_night_none.h5",
    "Weights of the model",
)
flags.DEFINE_string("image_process", "none", "Image process for image augmentation")
FLAGS = flags.FLAGS


def main(_):

    correspondance = yaml.safe_load(
        open("./improved_nightdrive/pipeline/correspondance.yml", "r")
    )
    new_classes = len(np.unique(list(correspondance.values())))

    default_config = {
        "model_name": FLAGS.model_name,
        "image_size": FLAGS.image_size,
        "intermediate_size": FLAGS.intermediate_size,
        "batch_size": FLAGS.batch_size,
        "num_classes": 19,
        "new_classes": new_classes,
        "image_process": FLAGS.image_process,
    }

    model = make_model(default_config)
    model.load_weights(FLAGS.model_weights)

    x_dir_path = FLAGS.dataset_x_path
    y_dir_path = FLAGS.dataset_y_path

    metrics = [ClassMeanIOU(new_classes), MeanIOU(new_classes)]
    preprocesses = [
        Resize(default_config["intermediate_size"]),
        ReClass(),
    ]
    eva = Evaluation(
        model,
        19,
        "cce",
        x_dir_path,
        y_dir_path,
        metrics,
        preprocesses,
        10,
        config=default_config,
    )

    m = eva.evaluate()
    res_str = f"mIOU : {m[1]}\n"
    for (p, n) in zip(
        m[0], ["Route", "Obstacles", "Panneaux", "Usagers fragiles", "Usagers"]
    ):
        res_str += n
        res_str += " "
        res_str += f"{p:.2f} | "
    print("\n", res_str, "\n")


if __name__ == "__main__":
    app.run(main)
