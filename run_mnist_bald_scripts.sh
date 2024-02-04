#!/bin/bash 

python3 run_image_classification_epig_40k.py --seed 3 --config configs/mnist_bald.yaml --device 'cpu'
python3 run_image_classification_epig_40k.py --seed 4 --config configs/mnist_bald.yaml --device 'cpu'
python3 run_image_classification_epig_40k.py --seed 5 --config configs/mnist_bald.yaml --device 'cpu'
python3 run_image_classification_epig_40k.py --seed 7 --config configs/mnist_bald.yaml --device 'cpu'
python3 run_image_classification_epig_40k.py --seed 9 --config configs/mnist_bald.yaml --device 'cpu'