from absl import app, flags
from datetime import datetime
import os
import yaml

import numpy as np

from improved_nightdrive.pipeline.metric import MeanIOU
from improved_nightdrive.pipeline.pipeline import Training
from improved_nightdrive.pipeline.preprocess import (
    AddNoise,
    EqualizeHistogram,
    FastClahe,
    GammaProcess,
    LogProcess,
    RandomCrop,
    RandomFlip,
    ReClass,
    Resize,
)
from improved_nightdrive.segmentation.models import make_model

flags.DEFINE_string(
    "model_name",
    "unetmobilenetv2",
    "Available models : [unetmobilenetv2, deeplabv3, unetmobilenetv2_big]",
)
flags.DEFINE_integer("image_size", 224, "Size of the square input for cropping")
flags.DEFINE_multi_integer("intermediate_size", (225, 400), "Size before cropping")
flags.DEFINE_integer("n_epochs", 50, "Number of epochs")
flags.DEFINE_integer("batch_size", 10, "Size of the batches")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
flags.DEFINE_string("dataset", "night_only", "Dataset")
flags.DEFINE_float("noise_factor", 1e-2, "Noise factor for image augmentation")
flags.DEFINE_string("image_process", "none", "Image process for image augmentation")
flags.DEFINE_float("gamma_factor", 0.5, "Arg for gamma process")
flags.DEFINE_float("log_factor", 0.5, "Arg for log process")
FLAGS = flags.FLAGS

x_dir_path_day = "./BDD100K/bdd100k/day/images/"
y_dir_path_day = "./BDD100K/bdd100k/day/labels/"

x_dir_path_night = "/media/raffaelbdl/T7/BDD100K/bdd100k/night/images/"
y_dir_path_night = "/media/raffaelbdl/T7/BDD100K/bdd100k/night/labels/"

x_dir_path_both = "./BDD100K/bdd100k/images/10k/train/"
y_dir_path_both = "./BDD100K/bdd100k/labels/sem_seg/colormaps/train/"


def main(_):

    correspondance = yaml.safe_load(
        open("./improved_nightdrive/pipeline/correspondance.yml", "r")
    )
    new_classes = len(np.unique(list(correspondance.values())))

    config = {
        "model_name": FLAGS.model_name,
        "image_size": FLAGS.image_size,
        "intermediate_size": tuple(FLAGS.intermediate_size),
        "num_epochs": FLAGS.n_epochs,
        "batch_size": FLAGS.batch_size,
        "learning_rate": FLAGS.learning_rate,
        "dataset": FLAGS.dataset,
        "num_classes": 19,
        "new_classes": new_classes,
        "noise_factor": FLAGS.noise_factor,
        "image_process": FLAGS.image_process,
        "gamma_factor": FLAGS.gamma_factor,
        "log_factor": FLAGS.log_factor,
    }

    train_name = config["model_name"] + "_" + config["dataset"]
    if config["image_process"] is not None:
        train_name += "_"
        train_name += config["image_process"]
    train_name += "_"
    train_name += str(datetime.now()).replace(" ", "").replace(":", "").replace(".", "")

    if not os.path.isdir(os.path.join("./results/", train_name)):
        os.makedirs(os.path.join("./results/", train_name, "evolution/"))
        os.makedirs(os.path.join("./results/", train_name, "models/"))

    model = make_model(config)
    model.summary()

    metrics = [
        MeanIOU(
            config["new_classes"]
            if config["new_classes"] > 0
            else config["num_classes"]
        ),
    ]

    preprocesses = [
        AddNoise(config["noise_factor"]),
        Resize(config["intermediate_size"]),
        RandomCrop(config["image_size"]),
        RandomFlip(),
        ReClass(),
    ]

    if config["image_process"] == "gamma_process":
        preprocesses.append(
            GammaProcess(
                p=[
                    config["gamma_factor"],
                    config["gamma_factor"],
                    config["gamma_factor"],
                ]
            )
        )
    elif config["image_process"] == "fast_clahe":
        preprocesses.append(FastClahe())
    elif config["image_process"] == "log_process":
        preprocesses.append(LogProcess(config["log_factor"]))
    elif config["image_process"] == "equalize_histogram":
        preprocesses.append(EqualizeHistogram())
    else:
        pass

    model_save_path = os.path.join("./results/", train_name, "models/")

    callbacks = []

    if config["dataset"] == "day_only":
        x_dir_path = x_dir_path_day
        y_dir_path = y_dir_path_day
    elif config["dataset"] == "night_only":
        x_dir_path = x_dir_path_night
        y_dir_path = y_dir_path_night
    elif config["dataset"] == "both":
        x_dir_path = x_dir_path_both
        y_dir_path = y_dir_path_both

    T = Training(
        model=model,
        num_classes=config["num_classes"],
        loss="wcce",
        x_dir_path=x_dir_path,
        y_dir_path=y_dir_path,
        metrics=metrics,
        preprocesses=preprocesses,
        callbacks=callbacks,
    )

    T.train(
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        save_model_bool=True,
        save_name=model_save_path,
        lr=config["learning_rate"],
    )


if __name__ == "__main__":
    app.run(main)