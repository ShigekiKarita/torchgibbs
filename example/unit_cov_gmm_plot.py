import torch
from matplotlib import pyplot

from torchgibbs.gibbs_sampler import UnitCovGaussianMixture

use_cuda = False
n = 10
x1 = torch.randn(n, 2) + torch.FloatTensor([[0.0, 5.0]])
x2 = torch.randn(n, 2) + torch.FloatTensor([[5.0, 0.0]])
x3 = torch.randn(n, 2) + torch.FloatTensor([[0.0, -5.0]])
for x in [x1, x2, x3]:
    pyplot.scatter(x[:, 0].numpy(), x[:, 1].numpy())

xs = torch.cat([x1, x2, x3], dim=0)

gmm = UnitCovGaussianMixture(2, 3)
if use_cuda:
    gmm.cuda()
    xs = xs.cuda()
gmm.fit(xs, 10)
for m in gmm.means.data:
    pyplot.scatter(m[0], m[1])

pyplot.show()