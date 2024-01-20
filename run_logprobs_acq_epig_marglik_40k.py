from math import sqrt
import logging
import torch
import wandb
import numpy as np
#from dotenv import load_dotenv
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
logging.basicConfig(format='[%(filename)s:%(lineno)s]%(levelname)s: %(message)s', level=logging.INFO)

from laplace.curvature import AsdlGGN
from laplace.utils import normal_samples

from active_learning.utils import TensorDataLoader, set_seed, CIFAR10_transform, dataset_to_tensors
from active_learning.active_learners import LaplaceActiveLearner, EnsembleActiveLearner, MoLaplaceActiveLearner
from active_learning.active_dataset import ActiveDataset


from src.math import count_correct, logmeanexp, nll_loss_from_probs
from src.trainers.base import LogProbsTrainer
from src.utils import Dictionary
from src.uncertainty import (
    bald_from_logprobs,
    bald_from_probs,
    epig_from_logprobs,
    epig_from_logprobs_using_matmul,
    epig_from_logprobs_using_weights,
    epig_from_probs,
    epig_from_probs_using_matmul,
    epig_from_probs_using_weights,
    marginal_entropy_from_logprobs,
    marginal_entropy_from_probs,
)

import hydra
from hydra import compose, initialize
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
import yaml

from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn.functional import log_softmax

def estimate_epig(
    m_model, loader: DataLoader, target_inputs: Tensor, use_matmul: bool
) -> Dictionary:
    #self.eval_mode()
    scores = Dictionary()

    for inputs, _ in loader:
        epig_scores = estimate_epig_minibatch(m_model, inputs, target_inputs, use_matmul)  # [B,]
        scores.append({"epig": epig_scores.cpu()})

    return scores.concatenate()

def estimate_epig_minibatch(#logprobs_trainer
    m_model, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
) -> Tensor:
    combined_inputs = torch.cat((inputs, target_inputs))  # [N + N_t, ...]
#     logprobs = self.conditional_predict(
#         combined_inputs, self.n_samples_test, independent=False
#     )  # [N + N_t, K, Cl]

    #our code istead of the function from above    
    n_samples = 150
    generator = None
    try:
        f_mu, f_var = m_model.la._glm_predictive_distribution(combined_inputs)
        f_samples = normal_samples(f_mu, f_var, n_samples, generator)
        features_aligned = torch.swapaxes(f_samples, 0, 1)
        logprobs = log_softmax(features_aligned, dim=-1)
    except:
        f_mu, f_var = m_model.la._glm_predictive_distribution(combined_inputs)
        f_var = torch.diagonal(f_var, dim1=1, dim2=2)
        f_samples = normal_samples(f_mu, f_var, n_samples, generator)
        features_aligned = torch.swapaxes(f_samples, 0, 1)
        logprobs = log_softmax(features_aligned, dim=-1)

    epig_fn = epig_from_logprobs_using_matmul if use_matmul else epig_from_logprobs
    return epig_fn(logprobs[: len(inputs)], logprobs[len(inputs) :])  # [N,]



# def estimate_epig_minibatch(#ProbsTrainer
#         m_model, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
#     ) -> Tensor:
#     _inputs = torch.cat((inputs, target_inputs))  # [N + N_t, ...]

#     n_samples = 150
#     generator = None
#     try:
#         f_mu, f_var = m_model._glm_predictive_distribution(combined_inputs)
#         f_samples = normal_samples(f_mu, f_var, n_samples, generator)
#         probs = torch.swapaxes(f_samples, 0, 1)
#     except:
#         f_mu, f_var = m_model._glm_predictive_distribution(combined_inputs)
#         f_var = torch.diagonal(f_var, dim1=1, dim2=2)
#         f_samples = normal_samples(f_mu, f_var, n_samples, generator)
#         probs = torch.swapaxes(f_samples, 0, 1)
    
#         epig_fn = epig_from_probs_using_matmul if use_matmul else epig_from_probs
#         return epig_fn(probs[: len(inputs)], probs[len(inputs) :])  # [N,]




def main(seed, dataset, n_init, n_max, optimizer, lr, lr_min, n_epochs, batch_size, method, approx, lr_hyp, lr_hyp_min,
         n_epochs_burnin, marglik_frequency, n_hypersteps, device, data_root, use_wandb, random_acquisition,
         early_stopping, last_layer, n_components, download_data, acquisition_method):
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

    train_dataset = ds_cls(data_root, train=True, download=download_data, transform=transform)
    test_dataset = ds_cls(data_root, train=False, download=download_data, transform=transform)
    x_original, y_original = dataset_to_tensors(train_dataset, device=device)
    x_test, y_test = dataset_to_tensors(test_dataset, device=device)
    test_loader = TensorDataLoader(x_test, y_test, batch_size=batch_size)

    set_seed(seed)
    
    classes = torch.unique(y_original)
    ixs_train = list()
    for c in classes.cpu().numpy():
        ixs_train.append(np.random.choice(np.where(y_original.cpu() == c)[0], int(40000/len(classes)), replace=False))
    ixs_train_np = np.array(ixs_train).reshape(1,-1).squeeze()
    x = x_original[ixs_train_np].detach().clone().to(device)
    y = y_original[ixs_train_np].detach().clone().to(device)
    
    print(f"Pool shape : {x.shape}")
    
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="main", overrides=["data=mnist/curated_pool", "experiment_name=mnist_curated", "acquisition.objective=epig"])
    rng = call(cfg.rng)
    data = instantiate(cfg.data, rng=rng)
    data.torch()
    data.to(device)
    
    classes = torch.unique(y_original)
    
    if acquisition_method == 'epig':
        ixs_targets = list()
        for c in classes.cpu().numpy():
            ixs_targets.append(np.random.choice(np.where(y_original.cpu() == c)[0], 
                                                int(10000/len(classes)), replace=False))
        ixs_targets_np = np.array(ixs_targets).reshape(1,-1).squeeze()
        X_targets = x_original[ixs_targets_np].detach().clone().to(device)
        y_targets = y_original[ixs_targets_np].detach().clone().to(device)
    else:
        X_targets = None
        y_targets = None

    # Set up model and initial training.
    dataset = ActiveDataset(x, y, n_init=n_init, stratified=True)
    if method == 'laplace':
        learner = LaplaceActiveLearner(
            input_size=input_size, output_size=10, likelihood='classification', model=model, double=False,
            device=device, lr=lr, lr_min=lr_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps,
            marglik_frequency=marglik_frequency, lr_hyp=lr_hyp, lr_hyp_min=lr_hyp_min,
            last_layer=last_layer, n_epochs_burnin=n_epochs_burnin, optimizer=optimizer,
            laplace=approx, backend=AsdlGGN, early_stopping=early_stopping
        )
    elif method == 'mola':
        learner = MoLaplaceActiveLearner(
            input_size=input_size, output_size=10, likelihood='classification', model=model, double=False,
            device=device, lr=lr, lr_min=lr_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps,
            marglik_frequency=marglik_frequency, lr_hyp=lr_hyp, lr_hyp_min=lr_hyp_min,
            last_layer=last_layer, n_epochs_burnin=n_epochs_burnin, optimizer=optimizer,
            laplace=approx, backend=AsdlGGN, early_stopping=early_stopping, n_components=n_components
        )
    elif method == 'ensemble':
        learner = EnsembleActiveLearner(
            input_size=input_size, output_size=10, likelihood='classification', model=model, double=False,
            device=device, lr=lr, lr_min=lr_min, n_epochs=n_epochs, optimizer=optimizer, backend=AsdlGGN,
            n_components=n_components)
    else:
        raise ValueError('Invalid active learner.')
    learner.fit(dataset.get_train_loader(batch_size=batch_size))
    # evaluate model on test set
    test_ll = learner.log_lik(test_loader)
    test_ll_bayes = learner.log_lik_bayes(test_loader)
    acc = learner.accuracy(test_loader)
    logging.info(f'Initial test log-likelihood: {test_ll:.4f}')
    logging.info(f'Initial test Bayes log-likelihood: {test_ll_bayes:.4f}')
    logging.info(f'Initial accuracy: {acc*100:.2f}%')
    if use_wandb:
        wandb.log({'test/ll': test_ll, 'test/ll_bayes': test_ll_bayes, 'test/acc': acc}, step=n_init)

    for i in range(n_init+1, min(len(x), n_max)):
        # acquire new data point
        if random_acquisition:
            dataset.add_ix(np.random.choice(dataset.not_ixs, 1)[0])
        elif acquisition_method == "bald":
            bald_scores = learner.bald(dataset.get_pool_loader(batch_size=batch_size))
            dataset.add_ix(dataset.not_ixs[torch.argmax(bald_scores).item()])
        elif acquisition_method == 'epig':
            print("Starting epig acquisition")
            #print(f"Target_inputs_batch_size: {len(target_inputs)}")
            target_loader = TensorDataLoader(X_targets, y_targets, batch_size = 100, shuffle=True)
            target_inputs, _ = next(iter(target_loader))
            #pool_loader = data.get_loader("pool")
            if cfg.acquisition.epig_probs_target != None:
                print("Estimate epig using pool")
#                 scores = estimate_epig_using_pool(
#                     learner, pool_loader, cfg.acquisition.epig_probs_target, cfg.acquisition.epig_probs_adjustment, len(target_inputs)
#                 )
                print("Something went wrong")
            else:
                print("Using original estimate_epig function")
                scores = estimate_epig(learner, dataset.get_pool_loader(batch_size=batch_size), target_inputs, cfg.acquisition.epig_using_matmul)           
            scores = scores.numpy()
            scores = scores['epig']
            acquired_pool_inds = np.argmax(scores)
            dataset.add_ix(dataset.not_ixs[acquired_pool_inds])
        #here comment
        # retrain model with new data point
        learner.fit(dataset.get_train_loader())

        # evaluate model on test set
        test_ll_bayes = learner.log_lik_bayes(test_loader)
        test_ll = learner.log_lik(test_loader)
        acc = learner.accuracy(test_loader)

        logging.info(f'Test log-likelihood at {i}: {test_ll:.4f}')
        logging.info(f'Test Bayes log-likelihood at {i}: {test_ll_bayes:.4f}')
        logging.info(f'Test accuracy: {acc*100:.2f}%')
        # optionally save to wandb
        if use_wandb:
            if random_acquisition:
                wandb.log({'test/ll': test_ll, 'test/ll_bayes': test_ll_bayes, 'test/acc': acc}, step=i)
            elif acquisition_method == 'bald':
                hist = wandb.Histogram(bald_scores.detach().cpu().numpy())
                wandb.log({'test/ll': test_ll, 'test/ll_bayes': test_ll_bayes, 'test/acc': acc, 'bald': hist}, 
                          step=i, commit=False)
            elif acquisition_method == 'epig':
                wandb.log({'test/ll': test_ll, 'test/ll_bayes': test_ll_bayes, 'test/acc': acc}, step=i)


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
    parser.add_argument('--method', default='laplace', help='Method', choices=['laplace', 'ensemble', 'mola'])
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
    parser.add_argument('--acquisition_method', default='epig', choices=['epig', 'bald'])
    parser.add_argument('--config', nargs='+')
    set_defaults_with_yaml_config(parser, sys.argv)
    args = vars(parser.parse_args())
    config_files = args['config']
    args.pop('config')
    if args['use_wandb']:
        import uuid
        import copy
        tags = [args['dataset'], args['method']] 
        if args['random_acquisition']:
            tags = [args['dataset'], args['method'], "rndm"] 
            ### left random_acquisition to be consistent with prev configs
        else:
            tags.append(args["acquisition_method"])
        config = copy.deepcopy(args)
        comps = tags
        run_name = '-'.join(comps)
        run_name += '-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, str(args)))[:4]
        config['methodconf'] = '-'.join(comps)
        #load_dotenv()
        wandb.init(project='alla', entity='marglik-is-the-best', config=config, name=run_name, tags=tags)
    main(**args)
