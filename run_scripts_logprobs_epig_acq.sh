#!/bin/bash 

python3 run_acq_epig_marglik_40k.py --seed 3 --config configs/mnist_tuned_burning_epig.yaml
python3 run_acq_epig_marglik_40k.py --seed 4 --config configs/mnist_tuned_burning_epig.yaml
python3 run_acq_epig_marglik_40k.py --seed 5 --config configs/mnist_tuned_burning_epig.yaml
python3 run_acq_epig_marglik_40k.py --seed 7 --config configs/mnist_tuned_burning_epig.yaml
python3 run_acq_epig_marglik_40k.py --seed 9 --config configs/mnist_tuned_burning_epig.yaml