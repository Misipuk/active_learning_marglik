#!/bin/bash 

python3 run_image_classification_40k_epig_acq.py --seed 5 --config configs/mnist_tuned_burning_epig.yaml
python3 run_image_classification_40k_epig_acq.py --seed 7 --config configs/mnist_tuned_burning_epig.yaml
python3 run_image_classification_40k_epig_acq.py --seed 9 --config configs/mnist_tuned_burning_epig.yaml
python3 run_image_classification_40k_epig_acq.py --seed 11 --config configs/mnist_tuned_burning_epig.yaml
python3 run_image_classification_40k_epig_acq.py --seed 15 --config configs/mnist_tuned_burning_epig.yaml
