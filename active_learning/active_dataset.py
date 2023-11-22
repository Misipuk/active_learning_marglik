import numpy as np
import torch

from active_learning.utils import TensorDataLoader


class ActiveDataset:

    def __init__(self, x, y, n_init, stratified=False) -> None:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        assert len(x) == len(y)
        self.x, self.y = x, y
        self.n_data = len(x)
        if stratified:
            # Sample n_init/classes data points from each class in y
            classes = torch.unique(y)
            ixs = list()
            for c in classes.cpu().numpy():
                ixs.append(np.random.choice(np.where(y.cpu() == c)[0], int(n_init/len(classes)), replace=False))
            self._ixs = np.concatenate(ixs)
        else:
            self._ixs = np.random.choice(self.n_data, size=n_init, replace=False)

    @property
    def ixs(self):
        return self._ixs

    @property
    def not_ixs(self):
        return np.setdiff1d(np.arange(self.n_data), self._ixs)

    def add_ixs(self, ixs):
        assert np.all(np.isin(ixs, np.arange(self.n_data)))
        assert np.all(np.isin(ixs, self.not_ixs))
        self._ixs = np.concatenate([self._ixs, ixs])

    def add_ix(self, ix):
        assert ix in np.arange(self.n_data)
        assert ix in self.not_ixs
        self._ixs = np.concatenate([self._ixs, [ix]])

    def get_train_loader(self, batch_size=256):
        x, y = self.x[self._ixs], self.y[self._ixs]
        return TensorDataLoader(x, y, batch_size=batch_size, shuffle=True)

    def get_pool_loader(self, batch_size=256):
        x, y = self.x[self.not_ixs], self.y[self.not_ixs]
        return TensorDataLoader(x, y, batch_size=batch_size, shuffle=False)

    def __len__(self):
        return len(self._ixs)


if __name__ == '__main__':
    np.random.seed(123)
    x, y = np.random.randn(100, 10), np.random.randn(100, 1)
    ds = ActiveDataset(x, y, 10)
    print(len(ds))
    ds.add_ix(89)
    ds.add_ixs([10, 51])
    ds.add_ixs(np.array([11, 52]))
    print(len(ds.not_ixs), len(ds.ixs), len(ds.get_train_loader().dataset),
          len(ds.get_pool_loader().dataset))
    for x, y in ds.get_pool_loader():
        print(x.shape, y.shape)


