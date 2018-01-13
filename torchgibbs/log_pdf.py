import math
import torch


def dirichlet_log_pdf(pi, alpha):
    numel = torch.lgamma(alpha.sum(0)) + torch.sum(torch.log(pi) * (alpha - 1.0))
    denom = torch.sum(torch.lgamma(alpha))
    return numel - denom


def normal_log_pdf(xs, means, cov):
    n_batch, n_dim = xs.size()
    n_component = means.size(0)
    assert cov.nelement() == 1
    xs_ms = xs.unsqueeze(1) - means.unsqueeze(0)
    coeff = - n_dim * math.log(2 * math.pi) - math.log(cov)
    xms = xs_ms.view(n_batch * n_component, n_dim, 1)
    pdfs = coeff + (-0.5 * xms.transpose(1, 2).bmm(xms) / cov)
    return pdfs.view(n_batch, n_component)