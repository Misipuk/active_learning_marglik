from math import sqrt
import logging
import torch
import wandb
import numpy as np
#from dotenv import load_dotenv
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
logging.basicConfig(format='[%(filename)s:%(lineno)s]%(levelname)s: %(message)s', level=logging.INFO)

import hydra
from hydra import compose, initialize
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
import yaml

from laplace.curvature import AsdlGGN

from active_learning.utils import TensorDataLoader, set_seed, CIFAR10_transform, dataset_to_tensors
from active_learning.active_learners import LaplaceActiveLearner, EnsembleActiveLearner, MoLaplaceActiveLearner
from active_learning.active_dataset import ActiveDataset


def main(seed, dataset, n_init, n_max, optimizer, lr, lr_min, n_epochs, batch_size, method, approx, lr_hyp, lr_hyp_min,
         n_epochs_burnin, marglik_frequency, n_hypersteps, device, data_root, use_wandb, random_acquisition,
         early_stopping, last_layer, n_components, download_data):
    if dataset == 'mnist':
        transform = transforms.ToTensor()
        ds_cls = MNIST
        model = 'cnn'
        input_size = (1, 28)
    elif dataset == 'cifar10':
        transform = CIFAR10_transform
        ds_cls = CIFAR10
        model = 'resnet'
        input_size = (3, 32)
    set_seed(seed)
    train_dataset = ds_cls(data_root, train=True, download=download_data, transform=transform)
    test_dataset = ds_cls(data_root, train=False, download=download_data, transform=transform)
    x_original, y_original = dataset_to_tensors(train_dataset, device=device)
    x_test, y_test = dataset_to_tensors(test_dataset, device=device)
    test_loader = TensorDataLoader(x_test, y_test, batch_size=batch_size)
    
    
    
    classes = torch.unique(y_original)
    ixs_train = list()
    for c in classes.cpu().numpy():
        ixs_train.append(np.random.choice(np.where(y_original.cpu() == c)[0], int(40000/len(classes)), replace=False))
    ixs_train_np = np.array(ixs_train).reshape(1,-1).squeeze()
    x = x_original[ixs_train_np].detach().clone().to(device)
    y = y_original[ixs_train_np].detach().clone().to(device)
    
    print(f"Pool shape : {x.shape}")
    
    
    ixs_val = list()
    for c in classes.cpu().numpy():
        ixs_val.append(np.random.choice(np.where(y.cpu() == c)[0], int(60/len(classes)), replace=False))
    ixs_val_np = np.array(ixs_val).reshape(1,-1).squeeze()
    x_val= x[ixs_val_np].detach().clone().to(device)
    y_val = y[ixs_val_np].detach().clone().to(device)
    val_loader = TensorDataLoader(x_val, y_val, batch_size=batch_size)

    

#     # Set up model and initial training.
#     dataset = ActiveDataset(x, y, n_init=n_init, stratified=True)
#     if method == 'laplace':
#         learner = LaplaceActiveLearner(
#             input_size=input_size, output_size=10, likelihood='classification', model=model, double=False,
#             device=device, lr=lr, lr_min=lr_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps,
#             marglik_frequency=marglik_frequency, lr_hyp=lr_hyp, lr_hyp_min=lr_hyp_min,
#             last_layer=last_layer, n_epochs_burnin=n_epochs_burnin, optimizer=optimizer,
#             laplace=approx, backend=AsdlGGN, early_stopping=early_stopping
#         )
#     elif method == 'mola':
#         learner = MoLaplaceActiveLearner(
#             input_size=input_size, output_size=10, likelihood='classification', model=model, double=False,
#             device=device, lr=lr, lr_min=lr_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps,
#             marglik_frequency=marglik_frequency, lr_hyp=lr_hyp, lr_hyp_min=lr_hyp_min,
#             last_layer=last_layer, n_epochs_burnin=n_epochs_burnin, optimizer=optimizer,
#             laplace=approx, backend=AsdlGGN, early_stopping=early_stopping, n_components=n_components
#         )
#     elif method == 'ensemble':
#         learner = EnsembleActiveLearner(
#             input_size=input_size, output_size=10, likelihood='classification', model=model, double=False,
#             device=device, lr=lr, lr_min=lr_min, n_epochs=n_epochs, optimizer=optimizer, backend=AsdlGGN,
#             n_components=n_components)
#     else:
#         raise ValueError('Invalid active learner.')
#     learner.fit(dataset.get_train_loader(batch_size=batch_size))


    dataset = ActiveDataset(x, y, n_init=n_init, stratified=True)
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="main", overrides=["data=mnist/unbalanced_pool", "experiment_name=mnist_unbalanced", "acquisition.objective=bald"])
    device = 'cpu'
    rng = call(cfg.rng)
    data = instantiate(cfg.data, rng=rng)
    data.torch()
    data.to(device)
    model = instantiate(cfg.model, input_shape=data.input_shape, output_size=data.n_classes)
    model = model.to(device)

#     cfg.trainer["n_optim_steps_max"] = 5
#     print(cfg.trainer["n_optim_steps_max"])
    trainer = instantiate(cfg.trainer, model=model)
    
    print("Training started")
    train_log = trainer.train(
            train_loader=dataset.get_train_loader(batch_size=batch_size),
            val_loader=val_loader,
    )
    
    print("Test started")
    with torch.inference_mode():
        test_acc, test_loss = trainer.test(test_loader)
        
    logging.info(f'Initial test loss: {test_loss:.4f}')
    logging.info(f'Initial accuracy: {test_acc*100:.2f}%')
    if use_wandb:
        wandb.log({'test/ll': test_loss, 'test/acc': test_acc}, step=n_init)

#     print("BALD started")

#     scores = trainer.estimate_bald(val_loader)
#     scores = scores.numpy()
#     scores = scores['bald']
#     acquired_pool_inds = np.argmax(scores)

#     print(f"index: {acquired_pool_inds}")


    for i in range(n_init+1, min(len(x), n_max)):
        # acquire new data point
        if random_acquisition:
            dataset.add_ix(np.random.choice(dataset.not_ixs, 1)[0])
        else:
            print("BALD started")
            scores = trainer.estimate_bald(dataset.get_pool_loader(batch_size=128))
            scores = scores.numpy()
            scores = scores['bald']
            acquired_pool_inds = np.argmax(scores)
            dataset.add_ix(dataset.not_ixs[np.argmax(scores)])
        
        val_loader = TensorDataLoader(x_val, y_val, batch_size=batch_size)
        # retrain model with new data point
        train_log = trainer.train(
            train_loader=dataset.get_train_loader(batch_size=batch_size),
            val_loader=val_loader,
        )

        # evaluate model on test set
        with torch.inference_mode():
            test_acc, test_loss = trainer.test(test_loader)

        logging.info(f'Test loss at {i}: {test_loss:.4f}')
        logging.info(f'Accuracy: {test_acc*100:.2f}%')
        # optionally save to wandb
        if use_wandb:
            if random_acquisition:
                wandb.log({'test/ll': test_loss, 'test/acc': test_acc}, step=i)
            else:
                hist = wandb.Histogram(np.copy(scores))
                wandb.log({'test/ll': test_loss, 'test/acc': test_acc, 'bald': hist}, 
                          step=i, commit=False)


if __name__ == '__main__':
    import sys
    import argparse
    from arg_utils import set_defaults_with_yaml_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--n_init', default=20, type=int)
    parser.add_argument('--n_max', default=500, type=int)
    # optimization (general)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_min', default=1e-6, type=float, help='Cosine decay target')
    parser.add_argument('--n_epochs', default=250, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--method', default='laplace', help='Method', choices=['laplace', 'ensemble', 'mola', 'epig_model'])
    # marglik-specific
    parser.add_argument('--approx', default='kron', choices=['full', 'kron', 'diag'])
    parser.add_argument('--lr_hyp', default=0.1, type=float)
    parser.add_argument('--lr_hyp_min', default=0.01, type=float)
    parser.add_argument('--n_epochs_burnin', default=10, type=int)
    parser.add_argument('--marglik_frequency', default=10, type=int)
    parser.add_argument('--n_hypersteps', default=50, help='Number of steps on every marglik estimate (partial grad accumulation)', type=int)
    parser.add_argument('--early_stopping', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--last_layer', default=False, action=argparse.BooleanOptionalAction)
    # ensemble-specific
    parser.add_argument('--n_components', default=10, type=int)
    # others
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--download_data', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_wandb', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--random_acquisition', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--config', nargs='+')
    set_defaults_with_yaml_config(parser, sys.argv)
    args = vars(parser.parse_args())
    config_files = args['config']
    args.pop('config')
    if args['use_wandb']:
        import uuid
        import copy
        tags = [args['dataset'], args['method']]
        config = copy.deepcopy(args)
        if config_files is not None:
            comps = [c.split('/')[-1].split('.')[0] for c in config_files]
        else:
            comps = tags
        run_name = '-'.join(comps)
        run_name += '-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, str(args)))[:4]
        config['methodconf'] = '-'.join(comps)
        #load_dotenv()
        wandb.init(project='alla', entity='marglik-is-the-best', config=config, name=run_name, tags=tags)
    main(**args)
