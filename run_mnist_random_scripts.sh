#!/bin/bash 

python3 run_image_classification.py --seed 3 --random_acquisition True --config configs/mnist.yaml
python3 run_image_classification.py --seed 4 --random_acquisition True --config configs/mnist.yaml
python3 run_image_classification.py --seed 5 --random_acquisition True --config configs/mnist.yaml
python3 run_image_classification.py --seed 7 --random_acquisition True --config configs/mnist.yaml
python3 run_image_classification.py --seed 9 --random_acquisition True --config configs/mnist.yaml
