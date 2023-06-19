# DBVCE-Eval
This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), which is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion). We removed every component which is not necessary for this project and extended the codebase with our own code.

# Install
1. `git clone`
2. Install submodules `git submodule update --init --recursive`
3. Install dependencies using conda `conda env create -f environment.yml`
4. Download your desired Madry and UNET model and adapt `MODEL_BASE_PATH` in `classifiers/paths.py` accordingly.
   1. Madry: `cls_ResNet50_madry/l2_improved_3_ep.pt`
   2. UNET: `256x256_classifier.pt`, taken from the guided_diffusion repository paper.
   3. The other models will be downloaded automatically.
5. Download and extract the ImageNet dataset. In our repository, we used `/data/imagenet/`.

# Sampling
See `run.sh`, which contains the commands to run the sampling. The sampling is done in these steps:

1. For every image-target pair, sample the results guided by the classifier and calculate metrics.
3. Calculate oracle scores.
4. Deploy to wandb.