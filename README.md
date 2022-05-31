# Improved-Semantic-Segmentation-for-Night-Drive
Project that aims at improving semantic segmentation on driving images at night-time, using GAN and lightning correction methods.

## Segmentation
### Installation

```bash
git clone git@github.com:tboulet/Improved-Semantic-Segmentation-for-Night-Drive.git
cd Improved-Semantic-Segmentation-for-Night-Drive

pip install -r requirements.txt
```

### Train segmentation

All tasks can be performed from main_segmentation.py.

To start a training run, please change the directory paths in `./main_segmentation.py`, then in bash:

```bash
python main_segmentation.py --model_name <unetmobilenetv2|deeplabv3> --dataset <day_only|night_only|both>
```

You can vizualize the inference results in `./results`.

## ForkGAN
ForkGAN is an improved version of CycleGAN able of generating a night-time dataset.

### Dataset
We used the BDD100K dataset available here: [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu).

Here are the paths to install it:

`ForkGAN`

    |datasets
        |BDD100K
            |trainA (images de jour)
            |trainB (images de nuit)
            |testA (images de jour)
            |testB (images de nuit)

### Requirements
The model works on Tensorflow 1. The list of the required modules can be found in the file ForkGAN/requirements.txt.

### Training
The model can be trained by lauching the script `scripts/bdd_train.sh` from the `ForkGAN/` folder.

### Create a night-time dataset
A night-time dataset can be created by launching the script `scripts/bdd_process_train.sh` from the `ForkGAN/` folder. He will then be saved in the `Processed_datasets/trainA` folder.

A small-sized dataset can be created by launching the script `scripts/bdd_process_train.sh` from the `ForkGAN/` folder. He will then be saved in the `Processed_datasets/testA` folder.

### Calculation of the metrics
#### FID of a created dataset
The Fréchet inception distance of a dataset can be calculated using the script `fid_metric.py`:
    `python improved_nightdrive/fid/fid_metric.py` "chemin vers le dataset"
    
The results are stored in the `ForkGAN/datasets/BDD100K/fid_logs` folder.

## Lighting

We provide the users with a *gradio* interface to make it easier to experiment with the lighting process.

In order to start the app, please run in bash:

```bash
python -m improved_nightdrive.app
``` 

Then click on the localhost link given in the terminal, it will redirect you to an interface on your default browser.