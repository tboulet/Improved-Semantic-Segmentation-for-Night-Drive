import os
import yaml
import argparse
import numpy as np
from typing import Tuple
from improved_nightdrive.fid.fid_metric import FID
from improved_nightdrive.segmentation.models import make_model
from improved_nightdrive.pipeline.pipeline import Evaluation, Training
from improved_nightdrive.pipeline.preprocess import AddNoise, RandomCrop, RandomFlip, Resize
from improved_nightdrive.pipeline.metric import MeanIOU


parser = argparse.ArgumentParser()

# Datasets directory
parser.add_argument("--datasets_dir", type=str, default="ForkGAN/Processed_datasets/images")
## datasets_dir
##           |_ cyclegan.model-2
##                   |_ images
##                   |_ labels
##           |_ cyclegan.model-2002
##                   |_ images
##                   |_ labels

# Model config
parser.add_argument("--model_name", type=str, default="deeplabv3", choices=["unetmobilenetv2", "deeplabv3"])
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--intermediate_size", type=Tuple[int, int], default=(225, 400))
parser.add_argument("--num_classes", type=int, default=19)

# Compute FID ?
parser.add_argument("--compute_fid", type=bool, default=False)
parser.add_argument("--fid_encoder", type=str, default="Inception", help="Inception or path to weights of a DeeplabV3 model")

# Use a pretrained model
parser.add_argument("--load_pretrained_model", type=bool, default=False)
parser.add_argument("--dataset", type=str, default="day_only", help="Which dataset the pretrained model has been trained on")

# Training config if we decide to train the model on each dataset
parser.add_argument("--train_on_datasets", type=bool, default=False, choices=[True, False])
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-4)

args = parser.parse_args()

config = dict()
for config_name, config_value in args._get_kwargs():
    config[config_name] = config_value

correspondance = yaml.safe_load(open("./improved_nightdrive/pipeline/correspondance.yml", "r"))
new_classes = len(np.unique(list(correspondance.values())))
config["new_classes"] = new_classes


metrics = [
    MeanIOU(config['num_classes']),
]
if args.compute_fid:
    fid = FID(
        encoder=args.fid_encoder,
        batch_size=8,
        image_size=(2*args.image_size, args.image_size)
    )

train_preprocesses = [
    AddNoise(),
    Resize(config['intermediate_size']),
    RandomCrop(config['image_size']),
    RandomFlip()
]
val_preprocesses = [
    Resize(config['intermediate_size']),
    RandomCrop(config['image_size']),
    RandomFlip()
]

results_dir = "results/datasets_test"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


c = len(os.listdir(results_dir))
with open(os.path.join(results_dir, f"experiment{c}.yaml"), "w") as results_file:
    results_file.write(f"# Experiment {c}\n\nConfig:\n")
    results_file.write(f"\tdatasets_dir: {config['datasets_dir']}\n\n")

    results_file.write(f"\tModel config:\n")
    results_file.write(f"\t\tmodel_name: {config['model_name']}\n")
    results_file.write(f"\t\timage_size: {config['image_size']}\n")
    results_file.write(f"\t\tintermediate_size: {config['intermediate_size']}\n")
    results_file.write(f"\t\tnum_classes: {config['num_classes']}\n\n")

    if args.load_pretrained_model:
        results_file.write(f"\tModel pretrained on {config['dataset']}\n\n")

    if args.train_on_datasets:
        results_file.write(f"\tTrained on each dataset with config:\n")
        results_file.write(f"\t\tnum_epochs: {config['num_epochs']}\n")
        results_file.write(f"\t\tbatch_size: {config['batch_size']}\n")
        results_file.write(f"\t\tlearning_rate: {config['learning_rate']}\n")

    if args.compute_fid:
        results_file.write(f"\tFID encoder: {args.fid_encoder}")

    results_file.write("\n\nResults:\n")


for dataset in os.listdir(args.datasets_dir):
    if dataset == "labels":
        continue
    model = make_model(config)

    if args.load_pretrained_model:
        weights_path = f"resutls/sweep/{args.model_name}_{args.dataset}/models/"
        # c = len(os.listdir(weights_path))
        # weights_path += f"{args.model_name}_{c}"
        # model.load_weights(weights_path)

    x_dir_path = os.path.join(args.datasets_dir, dataset, "images")
    y_dir_path = os.path.join(args.datasets_dir, "labels")

    if args.compute_fid:
        fid_value = fid(x_dir_path)

    if args.train_on_datasets:
        T = Training(
            model=model,
            num_classes=config['num_classes'],
            loss='cce',
            x_dir_path=x_dir_path,
            y_dir_path=y_dir_path,
            metrics=metrics,
            preprocesses=train_preprocesses,
        )

        T.train(
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['learning_rate']
        )

    model_results = Evaluation(
        model=model,
        num_classes=config["num_classes"],
        loss="cce",
        x_dir_path=x_dir_path,
        y_dir_path=y_dir_path,
        metrics=metrics,
        preprocesses=val_preprocesses,
        batch_size=8
    ).evaluate()

    with open(os.path.join(results_dir, f"experiment{c}.yaml"), "a") as results_file:
        results_file.write(f"\t{dataset}:\n")
        if args.compute_fid:
            results_file.write(f"\t\tFID: {fid_value}\n")
        for metric, metric_value in zip(metrics, model_results):
            results_file.write(f"\t\t{metric.name}: {metric_value}\n")
        results_file.write("\n")
