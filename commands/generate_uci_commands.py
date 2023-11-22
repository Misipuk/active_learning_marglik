from os import listdir
from itertools import product
seeds = [1, 2, 3, 4, 5]
datasets = ['power-plant', 'wine-quality-red', 'concrete', 'energy']
base_config = 'configs/uci.yaml'
method_configs = ['configs/methods/' + f for f in listdir('configs/methods') if f.endswith('.yaml')]
acquisitions = ['--no-random_acquisition', '--random_acquisition']

for acq, seed, dataset, method_config in product(acquisitions, seeds, datasets, method_configs):
    print(f'run_uci_regression.py --seed {seed} --dataset {dataset} --config {base_config} {method_config} {acq}')

