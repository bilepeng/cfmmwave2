import torch
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions import Uniform
from torch.optim import Adam
from collections import deque
import numpy as np
import torch.nn.functional as F


def determine_kappa(support, previous_kappa):
    lr_kappa = 1e-2
    if len(support) < 10:
        return previous_kappa
    elif support[-1] >= np.mean(support):
        return previous_kappa + lr_kappa
    else:
        return F.relu(previous_kappa - lr_kappa / 2)


if __name__ == "__main__":
    # 3 users, 2 APs
    channel = torch.tensor([[1, 1], [0.9, 0.9], [0., 0.]])
    # channel = torch.tensor([[1, 1], [0.9, 0.9], [0., 0.]]) * 1e-2

    # p = torch.tensor([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]).requires_grad_()
    p = (torch.ones((3, 2)) * (-3)).requires_grad_()
    l = (torch.ones((3, 2)) * 6).requires_grad_()
    optimizer = Adam([p, l], 1e-3)

    kappa = torch.zeros(1)
    support_history = deque()

    total_iter = 8000
    for iter in range(total_iter):
        temperature = 1 - iter / total_iter
        support_history.append(torch.nn.functional.softplus(l).sum().detach().numpy())
        while len(support_history) > 50:
            support_history.popleft()
        kappa = determine_kappa(support_history, kappa)

        q = Uniform(p, p + torch.nn.functional.softplus(l) + 1e-5)
        qq = q.rsample()
        s = torch.nn.functional.gumbel_softmax(logits=p, tau=temperature, hard=False, dim=-2)

        loss = -(torch.log(1 + (s * channel).sum(dim=-1))).sum() + kappa * l.sum()
        loss.backward()
        optimizer.step()
        print(p)
        print(torch.nn.functional.softplus(l))
        print(s)
        print(-loss)
        print(kappa)
        print("===")
        pass
