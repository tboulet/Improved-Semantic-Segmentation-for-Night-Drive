
import argparse
import os

import wandb

from improved_nightdrive.pipeline.callback import InferOnImage, WandbCallback 
from improved_nightdrive.pipeline.metric import MeanIOU
from improved_nightdrive.pipeline.pipeline import Training
from improved_nightdrive.pipeline.preprocess import AddNoise, RandomCrop, RandomFlip, Resize
from improved_nightdrive.segmentation.models import make_model


if not os.path.isdir("./results/"):
    os.mkdir("./results/")
if not os.path.isdir("./results/sweep/"):
    os.makedirs("./results/sweep/")
if not os.path.isdir("./results/sweep/deeplabv3_day_only/"):
    os.makedirs("./results/sweep/deeplabv3_day_only/models/")
    os.makedirs("./results/sweep/deeplabv3_day_only/evolution/")
if not os.path.isdir("./results/sweep/deeplabv3_night_only/"):
    os.makedirs("./results/sweep/deeplabv3_night_only/evolution/")
    os.makedirs("./results/sweep/deeplabv3_night_only/models/")
if not os.path.isdir("./results/sweep/deeplabv3_both/"):
    os.makedirs("./results/sweep/deeplabv3_both/evolution/")
    os.makedirs("./results/sweep/deeplabv3_both/models/")
if not os.path.isdir("./results/sweep/unetmobilenetv2_day_only/"):
    os.makedirs("./results/sweep/unetmobilenetv2_day_only/evolution/")
    os.makedirs("./results/sweep/unetmobilenetv2_day_only/models/")
if not os.path.isdir("./results/sweep/unetmobilenetv2_night_only/"):
    os.makedirs("./results/sweep/unetmobilenetv2_night_only/evolution/")
    os.makedirs("./results/sweep/unetmobilenetv2_night_only/models/")
if not os.path.isdir("./results/sweep/unetmobilenetv2_both/"):
    os.makedirs("./results/sweep/unetmobilenetv2_both/evolution")
    os.makedirs("./results/sweep/unetmobilenetv2_both/models/")

    

default_config = {
    'model_name': 'unetmobilenetv2',
    'image_size': 224,
    'intermediate_size': (225, 400),
    'num_epochs': 50,
    'batch_size': 3,
    'learning_rate': 1e-4,
    'dataset': 'day_only',
    'num_classes': 19
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

default_config['model_name'] = args.model_name
default_config['dataset'] = args.dataset

wandb.init(config=default_config)
config = wandb.config

model = make_model(config)

metrics = [
    MeanIOU(config['num_classes']),
]

preprocesses = [
    AddNoise(),
    Resize(config['intermediate_size']),
    RandomCrop(config['image_size']),
    RandomFlip()
]

inference_example_path = "./ressources/examples/"

if config['model_name'] == 'deeplabv3':
    if config['dataset'] == 'day_only':
        inference_save_path = "./results/sweep/deeplabv3_day_only/evolution/"
        model_save_path = "./results/sweep/deeplabv3_day_only/models/deeplabv3_"
    elif config['dataset'] == 'night_only':
        inference_save_path = "./results/sweep/deeplabv3_night_only/evolution/"
        model_save_path = "./results/sweep/deeplabv3_night_only/models/deeplabv3_"
elif config['model_name'] == 'unetmobilenetv2':
    if config['dataset'] == 'day_only':
        inference_save_path = "./results/sweep/unetmobilenetv2_day_only/evolution/"
        model_save_path = "./results/sweep/unetmobilenetv2_day_only/models/unetmobilenetv2_"
    elif config['dataset'] == 'night_only':
        inference_save_path = "./results/sweep/unetmobilenetv2_night_only/evolution/"
        model_save_path = "./results/sweep/unetmobilenetv2_night_only/models/unetmobilenetv2_"

callbacks = [
    InferOnImage(
        img_paths=[
            os.path.join(inference_example_path, "img_day_1.jpg"),
            os.path.join(inference_example_path, "img_day_2.jpg"),
            os.path.join(inference_example_path, "img_night_1.jpg"),
        ],
        label_paths=[
            os.path.join(inference_example_path, "seg_day_1.png"),
            os.path.join(inference_example_path, "seg_day_2.png"),
            os.path.join(inference_example_path, "seg_night_1.png"),
        ],
        size=config['image_size'],
        intermediate_size=config['intermediate_size'],
        save_path=inference_save_path
    ),
    WandbCallback()
]

if config['dataset'] == 'day_only':
    # x_dir_path = "./BDD100K/bdd100k/day/images/"
    # y_dir_path = "./BDD100K/bdd100k/day/labels/"
    x_dir_path = '/media/raffaelbdl/T7/BDD100K/bdd100k/day/images/'
    y_dir_path = '/media/raffaelbdl/T7/BDD100K/bdd100k/day/labels/'
elif config['dataset'] == 'night_only':
    x_dir_path = "./BDD100K/bdd100k/night/images/"
    y_dir_path = "./BDD100K/bdd100k/night/labels/"
elif config['dataset'] == 'both':
    x_dir_path = "./BDD100K/bdd100k/images/10k/train/"
    y_dir_path = "./BDD100K/bdd100k/labels/sem_seg/colormaps/train/"


T = Training(
    model=model,
    num_classes=config['num_classes'],
    loss='cce',
    x_dir_path=x_dir_path,
    y_dir_path=y_dir_path,
    metrics=metrics,
    preprocesses=preprocesses,
    callbacks=callbacks
)

T.train(
    num_epochs=config['num_epochs'],
    batch_size=config['batch_size'],
    save_model_bool=True,
    save_name=model_save_path,
    lr=config['learning_rate']
)
