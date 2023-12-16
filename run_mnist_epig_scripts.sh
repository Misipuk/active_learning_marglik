#!/bin/bash 

python3 run_image_classification_epig_40k_drop_false.py --seed 4 --config configs/mnist_epig.yaml --device 'cpu'
python3 run_image_classification_epig_40k_drop_false.py --seed 5 --config configs/mnist_epig.yaml --device 'cpu'
python3 run_image_classification_epig_40k_drop_false.py --seed 7 --config configs/mnist_epig.yaml --device 'cpu'
python3 run_image_classification_epig_40k_drop_false.py --seed 9 --config configs/mnist_epig.yaml --device 'cpu'
python3 run_image_classification_epig_40k_drop_false.py --seed 11 --config configs/mnist_epig.yaml --device 'cpu'