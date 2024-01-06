#!/bin/bash 

python3 run_image_classification.py --seed 3 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post.yaml
python3 run_image_classification.py --seed 4 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post.yaml
python3 run_image_classification.py --seed 5 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post.yaml
python3 run_image_classification.py --seed 7 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post.yaml
python3 run_image_classification.py --seed 9 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post.yaml

python3 run_image_classification.py --seed 3 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post_ll.yaml
python3 run_image_classification.py --seed 4 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post_ll.yaml
python3 run_image_classification.py --seed 5 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post_ll.yaml
python3 run_image_classification.py --seed 7 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post_ll.yaml
python3 run_image_classification.py --seed 9 --random_acquisition --config configs/mnist.yaml configs/methods/laplace_post_ll.yaml