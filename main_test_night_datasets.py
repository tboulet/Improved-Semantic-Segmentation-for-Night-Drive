import os
import argparse
from typing import Tuple
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


metrics = [
    MeanIOU(config['num_classes']),
]

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

    results_file.write("\n\nResults:")


for dataset in os.listdir(args.datasets_dir):
    model = make_model(config)

    if args.load_pretrained_model:
        weights_path = f"resutls/sweep/{args.model_name}_{args.dataset}/models/"
        # c = len(os.listdir(weights_path))
        # weights_path += f"{args.model_name}_{c}"
        # model.load_weights(weights_path)

    x_dir_path = os.path.join(args.datasets_dir, dataset, "images")
    y_dir_path = os.path.join(args.datasets_dir, dataset, "labels")

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
        batch_size=16
    ).evaluate()

    with open(os.path.join(results_dir, f"experiment{c}"), "a") as results_file:
        results_file.write(f"\t{dataset}\n")
        for metric, metric_value in zip(metrics, model_results):
            results_file.write(f"\t\t{metric.name}: {metric_value}\n")
        results_file.write("\n")
