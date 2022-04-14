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
