# Improved-Semantic-Segmentation-for-Night-Drive
Project for improve semantic segmentation on driving images by night, using GAN and lightning methods.

## Getting started
### Installation

```bash
git clone git@github.com:tboulet/Improved-Semantic-Segmentation-for-Night-Drive.git
cd Improved-Semantic-Segmentation-for-Night-Drive

pip install -r requirements.txt
```

### Train segmentation

All tasks can be performed from main_segmentation.py

To start a training run, please change the directory paths in `./main_segmentation.py`, then in bash:

```bash
python main_segmentation.py --model_name <unetmobilenetv2|deeplabv3> --dataset <day_only|night_only|both>
```

You can vizualize the inference results in `./results`.

## ForkGAN
ForkGAN est l'amélioration de CycleGAN qui nous permet de générer un dataset de nuit

### Dataset
Nous avons utilisé le dataset BDD100K disponible ici : [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)

Il doit être installé ainsi:
ForkGAN
    |datasets
        |BDD100K
            |trainA (images de jour)
            |trainB (images de nuit)
            |testA (images de jour)
            |testB (images de nuit)

### Requirements
Le modèle fonctionne avec tensorflow 1. La liste des modules supplémentaires requis se situe dans le fichier ForkgGAN/requirements.txt

### Entrainement
Le modèle peut être entrainé en lançant le script scripts/bdd_train.sh depuis le dossier ForkGAN/

### Créer un dataset de nuit
Un dataset de nuit peut être créé en lançant le script scripts/bdd_process_train.sh depuis le dossier ForkGAN/. Il sera enregistré dans Processed_datasets/trainA.

Un petit dataset de nuit peut être créé en lançant le script scripts/bdd_process_train.sh depuis le dossier ForkGAN/. Il sera enregistré dans Processed_datasets/testA.

### Calcul des métriques
#### FID d'un dataset créé
La distance FID d'un dataset créé peut être calculée en lançant le script fid_metric.py:
    python improved_nightdrive/fid/fid_metric.py "chemin vers le dataset"
Les résultats sont stockés dans ForkGAN/datasets/BDD100K/fid_logs
