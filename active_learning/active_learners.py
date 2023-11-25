import numpy as np
import torch
from torch.distributions import Normal, Categorical, MixtureSameFamily
from copy import deepcopy

from laplace.curvature import AsdlGGN, BackPackInterface

from .models import MLP, LeNet, ResNet
from .marglik import marglik_optimization
from .utils import get_laplace_approximation, get_lllaplace_approximation


class ActiveLearner:

    def __init__(self, input_size, output_size, likelihood, model, double, device) -> None:
        self.likelihood = likelihood
        if model == 'mlp':
            self.model = MLP(input_size, width=100, depth=1, output_size=output_size).to(device)
        elif model == 'cnn':
            assert len(input_size) == 2
            in_channels, in_pixels = input_size
            self.model = LeNet(in_channels=in_channels, n_pixels=in_pixels, n_out=output_size).to(device)
        elif model == 'resnet':
            self.model = ResNet(depth=8, num_classes=output_size).to(device)
        if double:
            self.model = self.model.double()

    def fit(self, loader):
        raise NotImplementedError

    def log_lik(self, loader):
        raise NotImplementedError

    def log_lik_bayes(self, loader):
        raise NotImplementedError

    def bald(self, loader):
        raise NotImplementedError


class LaplaceActiveLearner(ActiveLearner):

    def __init__(self, input_size, output_size, likelihood, model, double, device, lr=1e-3, lr_min=1e-6,
                 n_epochs=250, n_hypersteps=50, marglik_frequency=10, lr_hyp=1e-1, lr_hyp_min=1e-2, last_layer=False,
                 n_epochs_burnin=10, optimizer='sgd', laplace='kron', backend=AsdlGGN, early_stopping=False) -> None:
        super().__init__(input_size, output_size, likelihood, model, double, device)
        self.lr = lr
        self.lr_min = lr_min
        self.n_epochs = n_epochs
        self.n_hypersteps = n_hypersteps
        self.marglik_frequency = marglik_frequency
        self.lr_hyp = lr_hyp
        self.lr_hyp_min = lr_hyp_min
        self.last_layer = last_layer
        self.optimizer = optimizer
        self.laplace = laplace
        self.backend = backend
        self.es = early_stopping
        self.n_epochs_burnin = n_epochs_burnin

    def fit(self, loader):
        self.model.reset_parameters()
        laplace_cls = get_laplace_approximation(self.laplace)
        la, _, margliks = marglik_optimization(
            self.model, train_loader=loader, likelihood=self.likelihood, n_epochs=self.n_epochs,
            n_hypersteps=self.n_hypersteps, lr=self.lr, lr_min=self.lr_min, lr_hyp=self.lr_hyp,
            lr_hyp_min=self.lr_hyp_min, marglik_frequency=self.marglik_frequency, optimizer=self.optimizer,
            laplace=laplace_cls, n_epochs_burnin=self.n_epochs_burnin, backend=self.backend, early_stopping=self.es,
            prior_prec_init=1e-4
        )
        if self.last_layer:
            laplace_cls = get_lllaplace_approximation(self.laplace)
            la = laplace_cls(self.model, self.likelihood, backend=self.backend, sigma_noise=la.sigma_noise)
            la.fit(loader)

        if self.n_epochs < self.n_epochs_burnin or self.last_layer:
            if self.likelihood == 'regression':
                sigma_2 = torch.mean(torch.cat([(self.model(x).detach() - y).square() for x, y in loader], dim=0))
                la.sigma_noise = torch.sqrt(sigma_2)
            la.optimize_prior_precision()

        if len(margliks) > 0:
            self.marglik = margliks[-1] / len(loader.dataset)
        else:
            self.marglik = None
        self.la = la

    def log_lik_bayes(self, loader, scale=1.):
        ll = 0
        for x, y in loader:
            if self.likelihood == 'regression':
                f_mu, f_var = self.la(x)
                dist = Normal(f_mu.squeeze() * scale, scale * torch.sqrt(f_var + self.la.sigma_noise ** 2).squeeze())
                ll += dist.log_prob(y.squeeze() * scale).sum()
            elif self.likelihood == 'classification':
                p = self.la(x, link_approx='mc')
                dist = Categorical(probs=p)
                ll += dist.log_prob(y).sum()
        return ll / len(loader.dataset)

    def log_lik(self, loader, scale=1.):
        ll = 0
        for x, y in loader:
            if self.likelihood == 'regression':
                f_mu = self.model(x)
                dist = Normal(f_mu.squeeze() * scale, scale * self.la.sigma_noise)
                ll += dist.log_prob(y.squeeze() * scale).sum()
            elif self.likelihood == 'classification':
                f = self.model(x)
                dist = Categorical(logits=f)
                ll += dist.log_prob(y).sum()
        return ll / len(loader.dataset)

    def bald(self, loader):
        balds = list()
        for x, _ in loader:
            if self.likelihood == 'regression':
                _, f_var = self.la._glm_predictive_distribution(x)
                f_var = f_var.squeeze()
                y_var = self.la.sigma_noise ** 2 * torch.ones_like(f_var)
                balds.append(torch.log(f_var + y_var) - torch.log(y_var))
            elif self.likelihood == 'classification':
                try:
                    p_samples = self.la.predictive_samples(x, n_samples=150)
                except:
                    p_samples = self.la.predictive_samples(x, n_samples=150, diagonal_output = True)
                H = Categorical(probs=p_samples.mean(dim=0)).entropy()
                exp_H = Categorical(probs=p_samples).entropy().mean(dim=0)
                balds.append(H - exp_H)
            if balds[-1].dim() == 0:
                balds[-1] = balds[-1].unsqueeze(0)
        return torch.cat(balds)

    def accuracy(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total


class EnsembleActiveLearner(ActiveLearner):

    def __init__(self, input_size, output_size, likelihood, model, double, device, lr=1e-3, lr_min=1e-6,
                 n_epochs=250, optimizer='sgd', n_components=5, backend=AsdlGGN) -> None:
        super().__init__(input_size, output_size, likelihood, model, double, device)
        self.lr = lr
        self.lr_min = lr_min
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.backend = backend
        models = []
        for i in range(n_components):
            model = deepcopy(self.model)
            model.reset_parameters()
            models.append(model)
        del self.model
        self.models = models

    def fit(self, loader):
        for i, model in enumerate(self.models):
            model.reset_parameters()
            _, model, _ = marglik_optimization(
                model, train_loader=loader, likelihood=self.likelihood, n_epochs=self.n_epochs,
                lr=self.lr, lr_min=self.lr_min, optimizer=self.optimizer, n_epochs_burnin=self.n_epochs+1,
                backend=self.backend, prior_prec_init=1e-4
            )
        if self.likelihood == 'classification':
            return
        # choose sigma
        sigmas = list()
        for model in self.models:
            sqerr = 0
            for x, y in loader:
                sqerr += (model(x) - y).square().sum()
            sigmas.append(torch.sqrt(sqerr / len(loader.dataset)).item())
        self.sigmas = sigmas

    def get_pred(self, x):
        preds = list()
        for model in self.models:
            preds.append(model(x).detach())
        return torch.stack(preds, dim=1)

    def log_lik(self, loader, scale=1.):
        ll = 0
        for x, y in loader:
            pred = self.get_pred(x)  # n, models, outputs
            if self.likelihood == 'regression':
                mu = pred.squeeze()
                sigmas = torch.from_numpy(np.array(self.sigmas)).to(mu.device).to(mu.dtype).reshape(1, mu.shape[-1])
                unif = Categorical(probs=torch.ones_like(mu) / mu.shape[-1])
                norm = Normal(mu * scale, sigmas * scale)
                dist = MixtureSameFamily(unif, norm)
                ll += dist.log_prob(scale * y.squeeze()).sum()
            elif self.likelihood == 'classification':
                probs = torch.softmax(pred, dim=-1).mean(dim=1)  #n, outputs=classes
                dist = Categorical(probs=probs)
                ll += dist.log_prob(y).sum()
        return ll / len(loader.dataset)

    def log_lik_bayes(self, loader, scale=1.):
        return self.log_lik(loader, scale)

    def bald(self, loader):
        balds = list()
        for x, _ in loader:
            pred = self.get_pred(x)
            if self.likelihood == 'regression':
                f_var, _ = torch.var_mean(pred, dim=1)
                f_var = f_var.squeeze()
                y_var = np.mean(self.sigmas) ** 2 * torch.ones_like(f_var)
                balds.append(torch.log(f_var + y_var) - torch.log(y_var))
            elif self.likelihood == 'classification':
                H = Categorical(probs=torch.softmax(pred, dim=-1).mean(dim=1)).entropy()
                exp_H = Categorical(probs=torch.softmax(pred, dim=-1)).entropy().mean(dim=1)
                balds.append(H - exp_H)
            if balds[-1].dim() == 0:
                balds[-1] = balds[-1].unsqueeze(0)
        return torch.cat(balds)

    def accuracy(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                outputs = torch.softmax(self.get_pred(x), dim=-1).mean(dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total


class MoLaplaceActiveLearner(ActiveLearner):

    def __init__(self, input_size, output_size, likelihood, model, double, device, lr=1e-3, lr_min=1e-6,
                 n_epochs=250, n_hypersteps=50, marglik_frequency=10, lr_hyp=1e-1, lr_hyp_min=1e-2, last_layer=False,
                 n_epochs_burnin=10, optimizer='sgd', laplace='kron', backend=AsdlGGN, early_stopping=False,
                 n_components=5) -> None:
        super().__init__(input_size, output_size, likelihood, model, double, device)
        self.lr = lr
        self.lr_min = lr_min
        self.n_epochs = n_epochs
        self.n_hypersteps = n_hypersteps
        self.marglik_frequency = marglik_frequency
        self.lr_hyp = lr_hyp
        self.lr_hyp_min = lr_hyp_min
        self.last_layer = last_layer
        self.optimizer = optimizer
        self.laplace = laplace
        self.backend = backend
        self.es = early_stopping
        self.n_epochs_burnin = n_epochs_burnin
        self.n_components = n_components
        models = []
        for i in range(n_components):
            model = deepcopy(self.model)
            model.reset_parameters()
            models.append(model)
        del self.model
        self.models = models
        self.las = None
        self.marglik = None
        self.sigmas = None

    def fit(self, loader):
        marglik = 0
        las = list()
        sigmas = list()
        for i, model in enumerate(self.models):
            model.reset_parameters()
            laplace_cls = get_laplace_approximation(self.laplace)
            la, _, margliks = marglik_optimization(
                model, train_loader=loader, likelihood=self.likelihood, n_epochs=self.n_epochs,
                n_hypersteps=self.n_hypersteps, lr=self.lr, lr_min=self.lr_min, lr_hyp=self.lr_hyp,
                lr_hyp_min=self.lr_hyp_min, marglik_frequency=self.marglik_frequency, optimizer=self.optimizer,
                laplace=laplace_cls, n_epochs_burnin=self.n_epochs_burnin, backend=self.backend, early_stopping=self.es,
                prior_prec_init=1e-4
            )
            if self.last_layer:
                laplace_cls = get_lllaplace_approximation(self.laplace)
                la = laplace_cls(model, self.likelihood, backend=self.backend, sigma_noise=la.sigma_noise)
                la.fit(loader)

            if self.n_epochs < self.n_epochs_burnin or self.last_layer:
                # find optimal sigma_noise
                if self.likelihood == 'regression':
                    sigma_2 = torch.mean(torch.cat([(model(x).detach() - y).square() for x, y in loader], dim=0))
                    la.sigma_noise = torch.sqrt(sigma_2)
                la.optimize_prior_precision()

            if len(margliks) > 0:
                marglik += margliks[-1] / len(loader.dataset) / self.n_components
            if self.likelihood == 'regression':
                sigmas.append(la.sigma_noise.item())
            las.append(la)
        self.marglik = marglik
        self.las = las
        self.sigmas = sigmas

    def get_pred(self, x):
        preds = list()
        for model in self.models:
            preds.append(model(x).detach())
        return torch.stack(preds, dim=1)

    def get_bayes_pred(self, x):
        preds = list()
        for la in self.las:
            # each returned tensor is (n_samples, n, n_outputs)
            preds.append(la.predictive_samples(x, n_samples=100))
        return torch.cat(preds, dim=0).transpose(0, 1)  # (n, n_samples * n_comps, n_outputs)

    def get_glm_pred(self, x):
        mu, var = list(), list()
        for la in self.las:
            pred = la._glm_predictive_distribution(x)
            mu.append(pred[0].squeeze())
            var.append(pred[1].squeeze())
        return torch.stack(mu, dim=1), torch.stack(var, dim=1)

    def log_lik(self, loader, scale=1.):
        ll = 0
        for x, y in loader:
            pred = self.get_pred(x)  # n, models, outputs
            if self.likelihood == 'regression':
                mu = pred.squeeze()
                sigmas = torch.from_numpy(np.array(self.sigmas)).to(mu.device).to(mu.dtype).reshape(1, mu.shape[-1])
                unif = Categorical(probs=torch.ones_like(mu) / mu.shape[-1])
                norm = Normal(mu * scale, sigmas * scale)
                dist = MixtureSameFamily(unif, norm)
                ll += dist.log_prob(y.squeeze() * scale).sum()
            elif self.likelihood == 'classification':
                probs = torch.softmax(pred, dim=-1).mean(dim=1)  #n, outputs=classes
                dist = Categorical(probs=probs)
                ll += dist.log_prob(y).sum()
        return ll / len(loader.dataset)

    def log_lik_bayes(self, loader, scale=1.):
        ll = 0
        for x, y in loader:
            if self.likelihood == 'regression':
                f_mu, f_var = self.get_glm_pred(x)
                sigmas = torch.from_numpy(np.array(self.sigmas)).to(f_mu.device).to(f_mu.dtype).reshape(1, f_mu.shape[-1])
                unif = Categorical(probs=torch.ones_like(f_mu) / f_mu.shape[-1])
                norm = Normal(f_mu * scale, torch.sqrt(f_var + sigmas**2) * scale)
                dist = MixtureSameFamily(unif, norm)
                ll += dist.log_prob(y.squeeze() * scale).sum()
            elif self.likelihood == 'classification':
                pred = self.get_bayes_pred(x)  # n, models, outputs with softmax
                probs = pred.mean(dim=1)
                dist = Categorical(probs=probs)
                ll += dist.log_prob(y).sum()
        return ll / len(loader.dataset)

    def bald(self, loader):
        balds = list()
        for x, _ in loader:
            # NOTE: only difference to Ensemble is here in `get_bayes_pred` instead of `get_pred`
            pred = self.get_bayes_pred(x)
            if self.likelihood == 'regression':
                f_var = torch.var(pred, dim=1).squeeze()
                y_var = np.mean(self.sigmas) ** 2 * torch.ones_like(f_var)
                balds.append(torch.log(f_var + y_var) - torch.log(y_var))
            elif self.likelihood == 'classification':
                H = Categorical(probs=pred.mean(dim=1)).entropy()
                exp_H = Categorical(probs=pred).entropy().mean(dim=1)
                balds.append(H - exp_H)
            if balds[-1].dim() == 0:
                balds[-1] = balds[-1].unsqueeze(0)
        return torch.cat(balds)

    def accuracy(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                outputs = torch.softmax(self.get_pred(x), dim=-1).mean(dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total

