#!/bin/bash 

python3 run_acq_epig_marglik_40k.py --seed 3 --config configs/mnist_tuned_burning_epig_batch.yaml
python3 run_acq_epig_marglik_40k.py --seed 5 --config configs/mnist_tuned_burning_epig_batch.yaml
python3 run_acq_epig_marglik_40k.py --seed 7 --config configs/mnist_tuned_burning_epig_batch.yaml
python3 run_acq_epig_marglik_40k.py --seed 9 --config configs/mnist_tuned_burning_epig_batch.yaml
python3 run_acq_epig_marglik_40k.py --seed 10 --config configs/mnist_tuned_burning_epig_batch.yaml