# Improved Semantic Segmentation for Night Drive
This project aims at improving semantic segmentation on driving images at night-time, using GAN and lightning correction methods.
It was done in the frame of a semester project at CentraleSupélec, and was proposed by a team from Valeo, a french automotive supplier.

The members of this project are Raffael Bolla Di Lorenzo, Timothé Boulet, Alexandre Herment, Thibault Le Sellier de Chezelles and Hédi Razgallah.

This project is contained in the package `improved_nightdrive`, from which every functionalities can be called by following the procedure below.

## Installation
Start by installing all the requirements :

```bash
pip install -r requirements.txt
```

## Segmentation
A few examples are given in order to make the use of the code easier.

### Get the dataset

You can download the dataset BDD10K on [Berkeley's Website](https://bdd-data.berkeley.edu/portal.html#download).

### Get dataset metrics

To compute the metrics of a dataset, please run this command:

```bash
python examples/dataset_metrics.py
```

Available flags are:
```bash
examples/dataset_metrics.py:
  --dataset_y_path: Dataset images
    (default: './BDD100K/bdd100k/day/labels/')
```

As of now, the metrics are the proportion of classes per pixel.


### Train segmentation

To start a training run, please run this command:

```bash
python examples/train_segmentation.py
```

Available flags are :
```bash
examples/train_segmentation.py:
  --batch_size: Size of the batches
    (default: '10')
    (an integer)
  --dataset_x_path: Dataset images
    (default: './BDD100K/bdd100k/day/images/')
  --dataset_y_path: Dataset images
    (default: './BDD100K/bdd100k/day/labels/')
  --gamma_factor: Arg for gamma process
    (default: '0.5')
    (a number)
  --image_process: Image process for image augmentation
    (default: 'none')
  --image_size: Size of the square input for cropping
    (default: '224')
    (an integer)
  --intermediate_size: Size before cropping;
    repeat this option to specify a list of values
    (default: '[225, 400]')
    (an integer)
  --learning_rate: Learning rate
    (default: '0.0005')
    (a number)
  --log_factor: Arg for log process
    (default: '0.5')
    (a number)
  --model_name: Available models : [unetmobilenetv2, deeplabv3, unetmobilenetv2_big]
    (default: 'unetmobilenetv2')
  --n_epochs: Number of epochs
    (default: '50')
    (an integer)
  --noise_factor: Noise factor for image augmentation
    (default: '0.01')
    (a number)
```

### Evaluate a model on a dataset

To evaluate a model, please run this command:

```bash
python examples/eval_segmentation.py
```

Available flags are :
```bash
examples/eval_segmentation.py:
  --batch_size: Size of the batches
    (default: '10')
    (an integer)
  --dataset_x_path: Dataset images
    (default: './BDD100K/bdd100k/day/images/')
  --dataset_y_path: Dataset images
    (default: './BDD100K/bdd100k/day/labels/')
  --image_process: Image process for image augmentation
    (default: 'none')
  --image_size: Size of the square input for cropping
    (default: '224')
    (an integer)
  --intermediate_size: Size before cropping;
    repeat this option to specify a list of values
    (default: '[225, 400]')
    (an integer)
  --model_name: Available models : [unetmobilenetv2, deeplabv3, unetmobilenetv2_big]
    (default: 'unetmobilenetv2')
  --model_weights: Weights of the model
    (default: './results/unet_night_none.h5')
```

## Generating a dataset

### Dataset 
We used the BDD100K dataset available here: [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) for training the GANs.

### ForkGAN
ForkGAN is an improved version of CycleGAN able of generating a night-time dataset. The official implementation can be found here : [https://github.com/zhengziqiang/ForkGAN](https://github.com/zhengziqiang/ForkGAN).

To use the ForkGAN
Here are the paths to install it:

`ForkGAN`

    |datasets
        |BDD100K
            |trainA (images de jour)
            |trainB (images de nuit)
            |testA (images de jour)
            |testB (images de nuit)

#### Requirements
The model works on Tensorflow 1. The list of the required modules can be found in the file ForkGAN/requirements.txt.

#### Training
The model can be trained by lauching the script `scripts/bdd_train.sh` from the `ForkGAN/` folder.

#### Create a night-time dataset
A night-time dataset can be created by launching the script `scripts/bdd_process_train.sh` from the `ForkGAN/` folder. He will then be saved in the `Processed_datasets/trainA` folder.

A small-sized dataset can be created by launching the script `scripts/bdd_process_train.sh` from the `ForkGAN/` folder. He will then be saved in the `Processed_datasets/testA` folder.

### CoMoGAN
We used the official implementation that can be found here : (https://github.com/cv-rits/CoMoGAN)[https://github.com/cv-rits/CoMoGAN]

### Weights and dataset
The weights and generated datasets are available on this link:

https://drive.google.com/file/d/18nYMU-7Eq6VIy-YQoRmn04OaLVCKwtlx/view?usp=sharing

## Compute the FID of a dataset
The Fréchet inception distance of a dataset can be calculated using the script `fid_metric.py`:
    `python improved_nightdrive/fid/fid_metric.py` "chemin vers le dataset"
    
The results are stored in the `ForkGAN/datasets/BDD100K/fid_logs` folder.

## Lighting

We provide the users with a *gradio* interface to make it easier to experiment with the lighting process.

In order to start the app, please run this command:

```bash
python -m improved_nightdrive.app
``` 

Then click on the localhost link given in the terminal, it will redirect you to an interface on your default browser.

## Disclaimer

This code was made in the frame of a school project. It has not been extensively tested and may comport contain a few bugs and errors.
