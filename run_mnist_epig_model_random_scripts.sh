#!/bin/bash 

python3 run_image_classification_epig_40k_drop_false.py --seed 4 --random_acquisition --config configs/mnist_epig.yaml --device 'cpu'
python3 run_image_classification_epig_40k_drop_false.py --seed 5 --random_acquisition --config configs/mnist_epig.yaml --device 'cpu'
python3 run_image_classification_epig_40k_drop_false.py --seed 7 --random_acquisition --config configs/mnist_epig.yaml --device 'cpu'
python3 run_image_classification_epig_40k_drop_false.py --seed 9 --random_acquisition --config configs/mnist_epig.yaml --device 'cpu'
python3 run_image_classification_epig_40k_drop_false.py --seed 11 --random_acquisition --config configs/mnist_epig.yaml --device 'cpu'
