import torch
from torch.nn import Module, Parameter

from torchgibbs.log_pdf import normal_log_pdf


class DistributionSampler(Module):
    def sample(self, *inputs):
        pass


class GaussianMixtureMultinomial(DistributionSampler):
    def __init__(self, means, cov=None):
        super().__init__()
        self.means = means
        self.cov = Parameter(torch.FloatTensor([1.0])) if cov is None else cov

    def sample(self, xs):
        assert xs.size(1) == self.means.size(1)
        pxs = normal_log_pdf(xs, self.means.data, self.cov.data).exp()  # [B, K]
        pks = pxs / pxs.sum(dim=1, keepdim=True)  # [B, K]
        return torch.multinomial(pks, 1).squeeze(1)  # [B]


def select_k(xs, ids, k):
    n_batch, n_dim = xs.size()
    assert ids.size() == (n_batch,)
    assert isinstance(k, int)
    mask = ids == k
    mask = mask.expand(n_dim, xs.size(0)).transpose(0, 1)
    return torch.masked_select(xs, mask).view(-1, n_dim)


class Gaussian(DistributionSampler):
    def __init__(self, cov=1.0):
        super().__init__()
        self.cov = cov

    def sample(self, xs, sampled_ks, k):
        x_k = select_k(xs, sampled_ks, k)  # [B, D]
        n_k = 0 if x_k.dim() == 0 else x_k.size(0)
        x_mean_k = xs.new([0]) if x_k.dim() == 0 else torch.mean(x_k, dim=0)
        return torch.normal(n_k / (n_k + 1) * x_mean_k, self.cov / (n_k + 1))


class GaussianPriorGaussian(DistributionSampler):
    def __init__(self, cov, mean_prior_mean, mean_prior_inv_cov_factor):
        super().__init__()
        self.cov = cov
        self.mean_mean = mean_prior_mean
        self.mean_inv_cov_factor = mean_prior_inv_cov_factor

    def sample(self, xs, sampled_ks, k):
        x_k = select_k(xs, sampled_ks, k)  # [B, D]
        n_k = 0 if x_k.dim() == 0 else x_k.size(0)
        x_mean_k = xs.new([0]) if x_k.dim() == 0 else torch.mean(x_k, dim=0)
        mean_mean = (n_k * x_mean_k + self.mean_inv_cov_factor * self.mean_mean) / \
                    (n_k + self.mean_inv_cov_factor)
        mean_cov = self.cov / (n_k + self.mean_inv_cov_factor)
        return torch.nomal(mean_mean, mean_cov)