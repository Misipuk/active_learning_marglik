from math import ceil
import numpy as np
import torch
from torchvision import transforms


from laplace import FullLaplace, KronLaplace, DiagLaplace
from laplace import FullLLLaplace, KronLLLaplace, DiagLLLaplace


class TensorDataLoader:
    """Combination of torch's DataLoader and TensorDataset for efficient batch
    sampling and adaptive augmentation on GPU.
    """

    def __init__(self, x, y, transform=None, transform_y=None, batch_size=500,
                 data_factor=1, shuffle=False):
        assert x.size(0) == y.size(0), 'Size mismatch'
        self.x = x
        self.y = y
        self.device = x.device
        self.data_factor = data_factor
        self.n_data = y.size(0)
        if batch_size < 0:
            self.batch_size = self.x.size(0)
        else:
            self.batch_size = batch_size
        self.n_batches = ceil(self.n_data / self.batch_size)
        self.shuffle = shuffle
        identity = lambda x: x
        self.transform = transform if transform is not None else identity
        self.transform_y = transform_y if transform_y is not None else identity

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(self.n_data, device=self.device)
            self.x = self.x[permutation]
            self.y = self.y[permutation]
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        x = self.transform(self.x[start:end])
        y = self.transform_y(self.y[start:end])
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return self.n_batches

    @property
    def dataset(self):
        return DatasetDummy(self.n_data * self.data_factor)


class SubsetTensorDataLoader(TensorDataLoader):

    def __init__(self, x, y, transform=None, transform_y=None, subset_size=500,
                 data_factor=1, detach=True):
        self.subset_size = subset_size
        super().__init__(x, y, transform, transform_y, batch_size=subset_size,
                         data_factor=data_factor, shuffle=True, detach=detach)
        self.n_batches = 1  # -> len(loader) = 1

    def __iter__(self):
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        sod_indices = np.random.choice(self.n_data, self.subset_size, replace=False)
        if self._detach:
            x = self.transform(self.x[sod_indices]).detach()
        else:
            x = self.transform(self.x[sod_indices])
        y = self.transform_y(self.y[sod_indices])
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return DatasetDummy(self.subset_size * self.data_factor)


class DatasetDummy:
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return int(self.N)


def dataset_to_tensors(dataset, indices=None, device='cuda'):
    if indices is None:
        indices = range(len(dataset))  # all
    xy_train = [dataset[i] for i in indices]
    x = torch.stack([e[0] for e in xy_train]).to(device)
    y = torch.stack([torch.tensor(e[1]) for e in xy_train]).to(device)
    return x, y


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def get_laplace_approximation(structure):
    if structure == 'full':
        return FullLaplace
    elif structure == 'kron':
        return KronLaplace
    elif structure == 'diag':
        return DiagLaplace
    else:
        raise ValueError()


def get_lllaplace_approximation(structure):
    if structure == 'full':
        return FullLLLaplace
    elif structure == 'kron':
        return KronLLLaplace
    elif structure == 'diag':
        return DiagLLLaplace
    else:
        raise ValueError()


cifar10_mean = (0.49139968, 0.48215841, 0.44653091)
cifar10_std = (0.24703223, 0.24348513, 0.26158784)


CIFAR10_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
