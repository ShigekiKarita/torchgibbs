import torch
from torch.nn import Module, Parameter

from torchgibbs.distribution_sampler import GaussianMixtureMultinomial, Gaussian
from torchgibbs.log_pdf import normal_log_pdf


class GibbsSampler(Module):
    def log_joint_prob(self, *inputs):
        pass

    def fit(self, xs, n_iter):
        pass


class UnitCovGaussianMixture(GibbsSampler):
    """
    k ~ Multi(bin_probs)
    x[i] ~ Normal(mean[k], I)
    mean[k] ~ Normal(mean_of_mean, cov_of_mean * I)
    """
    def __init__(self, n_dim, n_component):
        super().__init__()
        self.n_dim = n_dim
        self.n_component = n_component
        self.means = Parameter(torch.randn(n_component, n_dim))
        self.mult = GaussianMixtureMultinomial(self.means)
        self.gauss = Gaussian()
        self.mean_inv_cov = 1.0
        self.mean_mean = Parameter(torch.zeros(n_dim))
        self.bin_probs = Parameter(torch.ones(n_component) / n_component)

    def assert_inputs(self, xs, ids=None):
        assert xs.dim() == 2
        assert xs.size(1) == self.n_dim
        if ids is not None:
            assert ids.size() == (xs.size(0),)

    def log_joint_prob(self, xs, ids):
        self.assert_inputs(xs, ids)
        px = normal_log_pdf(xs, self.means.data, 1.0).exp()
        px_k = px[torch.arange(0, xs.size(0), out=xs.new().long()), ids]
        log_pxz = torch.sum(px_k.log() + self.bin_probs.data.index_select(0, ids).log(), dim=0)
        p_mean = normal_log_pdf(self.means.data,
                                self.mean_mean.data.unsqueeze(0),
                                1.0 / self.mean_inv_cov)
        log_p_mean = torch.sum(p_mean, dim=0)
        return (log_pxz + log_p_mean)[0]

    def fit(self, xs, n_iter=1):
        self.assert_inputs(xs)
        for i in range(n_iter):
            sampled_ks = self.mult.sample(xs) # [B]
            for k in range(self.n_component):
                self.means.data[k] = self.gauss.sample(xs, sampled_ks, k)
            print(self.log_joint_prob(xs, sampled_ks))


class SphericalCovDirichletPriorGaussianMixture(GibbsSampler):
    """
    bin_probs ~ Dir(concentration)
    k ~ Multi(bin_probs)
    x[i] ~ Normal(mean[k], 1 / inv_cov * I)
    mean[k] ~ Normal(mean_of_mean, 1 / (inv_cov * mean_inv_cov_factor) * I)
    mean_inv_cov_factor ~ Gamma(mean_inv_cov_factor_shape, mean_inv_cov_scale)
    """

    def __init__(self, n_dim, n_component):
        super().__init__(n_dim, n_component)
        self.inv_cov = Parameter(torch.FloatTensor([1.0]))
        self.mean_inv_cov_factor = 1.0
        self.mean_inv_cov_factor_shape = 1.0
        self.mean_inv_cov_scale = 1.0

    def sample_k(self, xs):
        pxs = normal_log_pdf(xs, self.means.data, 1.0 / self.inv_cov).exp()  # [B, K]
        pxks = self.bin_probs.data.unsqueeze(0) * pxs  # [B, K]
        pks = pxks / pxks.sum(1, keepdim=True)
        return torch.multinomial(pks, 1).squeeze(1)  # [B]

    def sample_mean(self, xs, sampled_ks, k):
        x_k = self.select_k(xs, sampled_ks, k)  # [B, D]
        n_k = 0 if x_k.dim() == 0 else x_k.size(0)
        x_mean_k = xs.new([0]) if x_k.dim() == 0 else torch.mean(x_k, dim=0)
        mean_mean = (n_k * x_mean_k + self.mean_inv_cov_factor * self.mean_mean) / \
                    (n_k + self.mean_inv_cov_factor)
        mean_cov = self.cov / (n_k + self.mean_inv_cov_factor)
        return torch.nomal(mean_mean, mean_cov)

    def fit(self, xs, n_iter=1):
        self.assert_inputs(xs)
        for i in range(n_iter):
            pxs = normal_log_pdf(xs, self.means.data, 1.0).exp()  # [B, K]
            pxzs = self.bin_probs.data.unsqueeze(0) * pxs  # [B, K]
            pzs = pxzs / pxzs.sum(1, keepdim=True)