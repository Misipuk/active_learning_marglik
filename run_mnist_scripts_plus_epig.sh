#!/bin/bash 

python3 run_image_classification_40k_epig_acq.py --seed 15 --config configs/mnist_tuned_burning_epig.yaml
python3 run_image_classification_40k_epig_acq.py --seed 19 --config configs/mnist_tuned_burning_epig.yaml
python3 run_image_classification_40k_epig_acq.py --seed 21 --config configs/mnist_tuned_burning_epig.yaml
python3 run_image_classification_40k_epig_acq.py --seed 26 --config configs/mnist_tuned_burning_epig.yaml
python3 run_image_classification_40k_epig_acq.py --seed 30 --config configs/mnist_tuned_burning_epig.yaml