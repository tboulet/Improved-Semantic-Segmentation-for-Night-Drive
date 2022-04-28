
import argparse
import os
import yaml

import numpy as np
import wandb

from improved_nightdrive.pipeline.callback import InferOnImage, WandbCallback 
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
    Resize, 
    ReClass
)
from improved_nightdrive.segmentation.models import make_model


correspondance = yaml.safe_load(open("./improved_nightdrive/pipeline/correspondance.yml", "r"))
new_classes = len(np.unique(list(correspondance.values())))

default_config = {
    'model_name': 'unetmobilenetv2',
    'image_size': 224,
    'intermediate_size': (225, 400),
    'num_epochs': 50,
    'batch_size': 3,
    'learning_rate': 1e-4,
    'dataset': 'day_only',
    'num_classes': 19,
    'new_classes': new_classes,
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--dataset', type=str)
# Image Processing
parser.add_argument('--image_process', required=False, type=str)

args = parser.parse_args()

default_config['model_name'] = args.model_name
default_config['dataset'] = args.dataset
default_config['image_process'] = args.image_process if args.image_process else None

# wandb.init(config=default_config)
# config = wandb.config
config = default_config

train_name = config['model_name'] + '_' + config['dataset']
if config['image_process'] is not None:
    train_name += '_'
    train_name += config['image_process']
    
if not os.path.isdir(os.path.join("./results/", train_name)):
    os.makedirs(os.path.join("./results/", train_name, "evolution/"))
    os.makedirs(os.path.join("./results/", train_name, "models/"))

model = make_model(config)

metrics = [
    MeanIOU(config['new_classes'] if config['new_classes'] > 0 else config['num_classes']),
]

preprocesses = [
    AddNoise(),
    Resize(config['intermediate_size']),
    RandomCrop(config['image_size']),
    RandomFlip(),
    ReClass(),
]

if default_config['image_process'] == 'equalize_histogram': #BUG : broken
    preprocesses.append(EqualizeHistogram())
elif default_config['image_process'] == 'fast_clahe':
    preprocesses.append(FastClahe())
elif default_config['image_process'] == 'gamma_process':
    preprocesses.append(GammaProcess())
elif default_config['image_process'] == 'log_process':
    preprocesses.append(LogProcess())

inference_example_path = "./ressources/examples/"

inference_save_path = os.path.join("./results/", train_name, "evolution/")
model_save_path = os.path.join("./results/", train_name, "models/")

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
    # WandbCallback()
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
    lr=config['learning_rate'],
)
