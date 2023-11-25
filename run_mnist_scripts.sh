#!/bin/bash 

python3 run_image_classification.py --seed 3 --config configs/mnist.yaml
python3 run_image_classification.py --seed 4 --config configs/mnist.yaml
python3 run_image_classification.py --seed 5 --config configs/mnist.yaml
python3 run_image_classification.py --seed 7 --config configs/mnist.yaml
python3 run_image_classification.py --seed 9 --config configs/mnist.yaml