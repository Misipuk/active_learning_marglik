#!/bin/bash 

python3 run_logprobs_acq_epig_marglik_40k.py --seed 3 --random_acquisition --config configs/mnist_tuned_burning_epig.yaml
python3 run_logprobs_acq_epig_marglik_40k.py --seed 4 --random_acquisition --config configs/mnist_tuned_burning_epig.yaml
python3 run_logprobs_acq_epig_marglik_40k.py --seed 5 --random_acquisition --config configs/mnist_tuned_burning_epig.yaml
python3 run_logprobs_acq_epig_marglik_40k.py --seed 7 --random_acquisition --config configs/mnist_tuned_burning_epig.yaml
python3 run_logprobs_acq_epig_marglik_40k.py --seed 9 --random_acquisition --config configs/mnist_tuned_burning_epig.yaml