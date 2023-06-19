#!/bin/bash

# Working settings:
#    --classifier_class AlexNetClassifier\
#    --classifier_class AlexNetRandomInitClassifier\
#    --classifier_class MadryClassifier\
#    -- UnetEncoderClassifier
#    -- ConvNeXtLClassifier
# SIMCLRClassifier
# SwinTFLClassifier
# YOU CAN SET --classifier_path  to overwrite default classifier path
python -m scripts.classifier_sample \
    --data_path /data/imagenet/\
    --classifier_class ConvNeXtLClassifier\
    --classifier_perturbation 1\
    --model_path models/256x256_diffusion_uncond.pt\
    --use_fp16 True\
    --cone_projection True\
    --xzero_prediction True\
    --seed 1\
    --batch_size 50 --num_samples 50\
    --lp_custom 1.0 --lp_regularization_scale 0.15\
    --classifier_scale 0.1\
    --skip_timesteps 100\
    --init_image True

