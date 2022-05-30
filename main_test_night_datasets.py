import os
import yaml
import argparse
import numpy as np
from typing import Tuple
from improved_nightdrive.fid.fid_metric import FID
from improved_nightdrive.segmentation.models import make_model
from improved_nightdrive.pipeline.pipeline import Evaluation, Training
from improved_nightdrive.pipeline.preprocess import AddNoise, RandomCrop, RandomFlip, Resize, GaussianBlur, ReClass
from improved_nightdrive.pipeline.metric import MeanIOU, ClassMeanIOU


parser = argparse.ArgumentParser()

# Datasets directory
parser.add_argument("--dataset_dir", type=str, default="datas/day/")
# datasets_dir
#           |_ labels
#           |_ images

# Model config
parser.add_argument("--model_name", type=str, default="unetmobilenetv2", choices=["unetmobilenetv2", "deeplabv3"])
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--intermediate_size", type=Tuple[int, int], default=(225, 400))
parser.add_argument("--num_classes", type=int, default=19)

# Compute FID ?
parser.add_argument("--compute_fid", type=bool, default=False)
parser.add_argument("--fid_encoder", type=str, default="Inception", help="Inception or path to weights of a DeeplabV3 model")

# Use a pretrained model
parser.add_argument("--pretrained_model", default=None)

# Training config if we decide to train the model on dataset
parser.add_argument("--train_on_dataset", type=bool, default=False, choices=[True, False])
parser.add_argument("--test_dataset_path", type=str, default="datas/night/")
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)

args = parser.parse_args()

config = dict()
for config_name, config_value in args._get_kwargs():
    config[config_name] = config_value

correspondance = yaml.safe_load(open("./improved_nightdrive/pipeline/correspondance.yml", "r"))
new_classes = len(np.unique(list(correspondance.values())))
config["new_classes"] = new_classes
# config["num_classes"] = new_classes


metrics = [
    MeanIOU(config['new_classes']),
    ClassMeanIOU(config['new_classes']),
]
if args.compute_fid:
    fid = FID(
        encoder=args.fid_encoder,
        batch_size=8,
        image_size=(2*args.image_size, args.image_size),
        path_night_real_images="datas/night/images"
    )

train_preprocesses = [
    GaussianBlur(5, 1),
    AddNoise(0.1),
    Resize(config['intermediate_size']),
    RandomCrop(config['image_size']),
    RandomFlip(),
    ReClass(config["num_classes"]),
]
val_preprocesses = [
    Resize(config['intermediate_size']),
    RandomCrop(config['image_size']),
    ReClass(config["num_classes"]),
]

results_dir = "results/datasets_test"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


c = len(os.listdir(results_dir))
with open(os.path.join(results_dir, f"experiment{c}.yaml"), "w") as results_file:
    results_file.write(f"# Experiment {c}\n\nConfig:\n")
    results_file.write(f" dataset_dir: {config['dataset_dir']}\n\n")

    results_file.write(" Model config:\n")
    results_file.write(f"  model_name: {config['model_name']}\n")
    results_file.write(f"  image_size: {config['image_size']}\n")
    results_file.write(f"  intermediate_size: {config['intermediate_size']}\n")
    results_file.write(f"  num_classes: {config['num_classes']}\n\n")

    if args.pretrained_model is not None:
        results_file.write(f" pretrained weights: {config['pretrained_model']}\n\n")

    if args.train_on_dataset:
        results_file.write(" Trained on dataset with config:\n")
        results_file.write(f"  test_dataset: {config['test_dataset_path']}\n")
        results_file.write(f"  num_epochs: {config['num_epochs']}\n")
        results_file.write(f"  batch_size: {config['batch_size']}\n")
        results_file.write(f"  learning_rate: {config['learning_rate']}\n")

    if args.compute_fid:
        results_file.write(f" FID encoder: {args.fid_encoder}")

    results_file.write("\n\nResults:\n")


model = make_model(config)

if args.pretrained_model is not None:
    model.load_weights(args.pretrained_model)

x_dir_path = os.path.join(args.dataset_dir, "images")
y_dir_path = os.path.join(args.dataset_dir, "labels")

if args.compute_fid:
    fid_value = fid(x_dir_path)


if args.train_on_dataset:
    T = Training(
        model=model,
        num_classes=config['num_classes'],
        loss='wcce',
        x_dir_path=x_dir_path,
        y_dir_path=y_dir_path,
        metrics=metrics,
        preprocesses=train_preprocesses,
    )
    x_test_dir = os.path.join(args.test_dataset_path, "images")
    y_test_dir = os.path.join(args.test_dataset_path, "labels")
    E = Evaluation(
        model=model,
        num_classes=config["num_classes"],
        loss="wcce",
        x_dir_path=x_test_dir,
        y_dir_path=y_test_dir,
        metrics=metrics,
        preprocesses=val_preprocesses,
        batch_size=8
    )
    for i in range(config["num_epochs"]):

        train_res = T.train(
            num_epochs=1,
            batch_size=config['batch_size'],
            lr=config['learning_rate']
        )
        eval_res = E.evaluate()

        with open(os.path.join(results_dir, f"experiment{c}.yaml"), "a") as results_file:
            results_file.write(f" Epoch{i}:\n")
            if args.compute_fid:
                results_file.write(f"  FID: {fid_value}\n")
            for metric, metric_value in zip(metrics, train_res):
                results_file.write(f"  train_{metric.name}: {metric_value}\n")
            for metric, metric_value in zip(metrics, eval_res):
                results_file.write(f"  eval_{metric.name}: {metric_value}\n")
            results_file.write("\n")

else:
    x_test_dir = x_dir_path
    y_test_dir = y_dir_path

    model_results = Evaluation(
        model=model,
        num_classes=config["num_classes"],
        loss="wcce",
        x_dir_path=x_dir_path,
        y_dir_path=y_dir_path,
        metrics=metrics,
        preprocesses=val_preprocesses,
        batch_size=8
    ).evaluate()

    with open(os.path.join(results_dir, f"experiment{c}.yaml"), "a") as results_file:
        if args.compute_fid:
            results_file.write(f" FID: {fid_value}\n")
        for metric, metric_value in zip(metrics, model_results):
            results_file.write(f" {metric.name}: {metric_value}\n")
        results_file.write("\n")
