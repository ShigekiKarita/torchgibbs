from unittest import TestCase
import torch
from torchgibbs.gibbs_sampler import SphericalCovDirichletPriorGaussianMixture


def test_sample_k():
    n_dim = 2
    gmm = SphericalCovDirichletPriorGaussianMixture(n_dim, 3)
    # gmm = UnitCovGaussianMixture(n_dim, 3)
    n_batch = 10
    xs = torch.randn(n_batch, n_dim)
    ids = torch.zeros(n_batch)
    ids[0] = 1
    ids[2] = 1
    ids[-1] = 1
    ys = gmm.sample_mean(xs, ids, 1)
    assert torch.equal(ys[0], xs[0])
    assert torch.equal(ys[1], xs[2])
    assert torch.equal(ys[2], xs[-1])
