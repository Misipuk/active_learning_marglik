#!/bin/bash 

python3 run_acq_epig_marglik_40k.py --seed 3 --no-epig_use_logprobs --config configs/mnist_tuned_burning_epig.yaml
python3 run_acq_epig_marglik_40k.py --seed 4 --no-epig_use_logprobs --config configs/mnist_tuned_burning_epig.yaml
python3 run_acq_epig_marglik_40k.py --seed 5 --no-epig_use_logprobs --config configs/mnist_tuned_burning_epig.yaml
python3 run_acq_epig_marglik_40k.py --seed 7 --no-epig_use_logprobs --config configs/mnist_tuned_burning_epig.yaml
python3 run_acq_epig_marglik_40k.py --seed 9 --no-epig_use_logprobs --config configs/mnist_tuned_burning_epig.yaml