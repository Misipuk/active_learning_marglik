from copy import deepcopy
import numpy as np
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.distributions import Normal

from laplace import KronLaplace
from laplace.curvature import AsdlGGN

GB_FACTOR = 1024 ** 3


def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])


def get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device):
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    return log_prior_prec


def valid_performance(model, test_loader, likelihood, criterion, device):
    N = len(test_loader.dataset)
    perf = 0
    nll = 0
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        with torch.no_grad():
            f = model(X)
        if likelihood == 'classification':
            perf += (torch.argmax(f, dim=-1) == y).sum() / N
        elif likelihood == 'heteroscedastic_regression':
            perf += (y.squeeze() + 0.5 * f[:, 0] / f[:, 1]).square().sum() / N
        else:
            perf += (f - y).square().sum() / N
        nll += criterion(f, y) / len(test_loader)
    return perf.item(), nll.item()


def get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min):
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        return CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')


def get_model_optimizer(optimizer, model, lr, weight_decay=0):
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        # fixup parameters should have 10x smaller learning rate
        is_fixup = lambda param: param.size() == torch.Size([1])  # scalars
        fixup_params = [p for p in model.parameters() if is_fixup(p)]
        standard_params = [p for p in model.parameters() if not is_fixup(p)]
        params = [{'params': standard_params}, {'params': fixup_params, 'lr': lr / 10.}]
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')


def gradient_to_vector(parameters):
    return parameters_to_vector([e.grad for e in parameters])


def vector_to_gradient(vec, parameters):
    return vector_to_parameters(vec, [e.grad for e in parameters])


def marglik_optimization(model,
                         train_loader,
                         valid_loader=None,
                         likelihood='classification',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         sigma_noise_init=1.,
                         temperature=1.,
                         n_epochs=500,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='cos',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         lr_hyp_min=1e-1,
                         laplace=KronLaplace,
                         backend=AsdlGGN,
                         early_stopping=False,
                         gamma_hyp = False,
                         gamma_alpha_a = 1,
                         gamma_alpha_b = 1,
                         gamma_beta_a = 0.5,
                         gamma_beta_b = 0.5):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    likelihood : str
        'classification', 'regression', 'heteroscedastic_regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    sigma_noise_init : float
        initial observation noise (for regression only)
    temperature : float
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    n_epochs : int
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'adam' or 'sgd'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
    lr_hyp : float
        learning rate for hyperparameters (should be between 1e-3 and 1)
    laplace : Laplace
        type of Laplace approximation (Kron/Diag/Full)
    backend : Backend
        AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    stochastic_grad : bool
    independent : bool
        whether to use independent functional laplace
    single_output : bool
        whether to use single random output for functional laplace
    kron_jac : bool
        whether to use kron_jac in the backend

    Returns
    -------
    lap : Laplace
        lapalce approximation
    model : torch.nn.Module
    margliks : list
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    best_model_dict = None

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)
    hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='mean')
        sigma_noise = 1
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)
    else:
        raise ValueError()

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)

    n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * n_hypersteps
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    hyper_scheduler = CosineAnnealingLR(hyper_optimizer, n_steps, eta_min=lr_hyp_min)

    losses = list()
    valid_perfs = list()
    valid_nlls = list()
    margliks = list()
    best_marglik = np.inf
    
    if gamma_hyp:
        print(f"Prec Gamma {gamma_alpha_a, gamma_alpha_b}")
        alpha_dist = Gamma(gamma_alpha_a, gamma_alpha_b)
        beta_dist = Gamma(gamma_beta_a, gamma_beta_b)
        if std_hyp:
            print(f"Sigma noise Gamma {gamma_beta_a, gamma_beta_b}")

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict(epoch=epoch)

        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()

            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = 1 / temperature / (2 * sigma_noise.square())
            else:
                crit_factor = 1 / temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            delta = expand_prior_precision(prior_prec, model)

            f = model(X)

            theta = parameters_to_vector(model.parameters())
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()

            epoch_loss += loss.cpu().item() / len(train_loader)
            epoch_nll += criterion(f.detach(), y).item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            elif likelihood == 'heteroscedastic_regression':
                epoch_perf += (y.squeeze() + 0.5 * f[:, 0] / f[:, 1]).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()

        losses.append(epoch_loss)
        #logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf:.2f}; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr})
        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                if likelihood == 'regression':
                    def val_criterion(f, y):
                        assert f.shape == y.shape
                        log_lik = Normal(loc=f, scale=sigma_noise).log_prob(y)
                        return -log_lik.mean()
                else:
                    val_criterion = criterion
                val_perf, val_nll = valid_performance(model, valid_loader, likelihood, val_criterion, device)
                valid_perfs.append(val_perf)
                valid_nlls.append(val_nll)
                #logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf:.2f}; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})

        # only update hyperparameters every "Frequency" steps after "burnin"
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # 1. fit laplace approximation
        torch.cuda.empty_cache()

        sigma_noise = 1 if likelihood != 'regression' else torch.exp(log_sigma_noise)
        prior_prec = torch.exp(log_prior_prec)
        lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                        temperature=temperature, backend=backend)
        lap.fit(train_loader)
        # first optimize prior precision jointly with direct marglik grad
        margliks_local = list()
        for i in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            sigma_noise = None if likelihood != 'regression' else torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            
            if gamma_hyp:
                #prior_prec = torch.nan_to_num(prior_prec, nan=0.00001)
                if std_hyp:
                    #print("We are in std_hyp")
                    marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / N - alpha_dist.log_prob(prior_prec).sum() / N - beta_dist.log_prob(sigma_noise).sum() / N
                else:
                    #print("We are not in std_hyp")
                    marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / N - alpha_dist.log_prob(prior_prec).sum() / N
            else:
                marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / N
            
            marglik.backward()
            margliks_local.append(marglik.item())
            hyper_optimizer.step()
            hyper_scheduler.step()

        marglik = margliks_local[-1]

        if likelihood == 'regression':
            epoch_log['hyperparams/sigma_noise'] = torch.exp(log_sigma_noise.detach()).cpu().item()
        epoch_log['train/marglik'] = marglik
        margliks.append(marglik)
        del lap

        # early stopping on marginal likelihood
        if early_stopping and (margliks[-1] < best_marglik):
            best_model_dict = deepcopy(model.state_dict())
            best_precision = deepcopy(prior_prec.detach())
            best_sigma = 1 if likelihood != 'regression' else deepcopy(sigma_noise.detach())
            best_marglik = margliks[-1]

    if early_stopping and (best_model_dict is not None):
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
    else:
        sigma_noise = 1 if sigma_noise is None else sigma_noise

    lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                  temperature=temperature, backend=backend)
    lap.fit(train_loader)
    return lap, model, margliks
