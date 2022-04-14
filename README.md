# Improved-Semantic-Segmentation-for-Night-Drive
Project for improve semantic segmentation on driving images by night, using GAN and lightning methods.



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