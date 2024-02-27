import copy

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from torch.distributions import Uniform

# Debug
from torch.optim import Adam


class Gamma(nn.Module):
    def __init__(self, params, input_dim, output_dim, device="cpu", final_layer=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, params["gamma_feature_dim"] * 2, (1, 1)).to(device)
        # self.conv2 = nn.Conv3d(params["gamma_feature_dim"] * 4, params["gamma_feature_dim"] * 4, (1, 1, 1)).to(device)
        self.conv2 = nn.Conv2d(params["gamma_feature_dim"] * 2, output_dim, (1, 1)).to(device)
        self.ex = (torch.ones((params["num_users"], params["num_users"])) -
                   torch.eye(params["num_users"])).requires_grad_().to(device)
        self.n_cat1 = params["num_users"] - 1
        self.feature_dim = params["gamma_feature_dim"]
        self.num_users = params["num_users"]
        self.final_layer = final_layer

    def forward(self, gamma_input):
        # Input format: (data sample, feature, user, AP)
        def postprocess_layer(conv_output):
            features_c0 = conv_output[:, : self.feature_dim, :, :]
            features_c1 = self.ex @ conv_output[:, self.feature_dim: (self.feature_dim * 2), :, :] / self.n_cat1
            return torch.cat((features_c0, features_c1), 1)

        r = F.relu(self.conv1(gamma_input))
        r = postprocess_layer(r)
        # r = F.relu(self.conv2(r))
        # r = postprocess_layer(r)
        r = self.conv2(r)
        if not self.final_layer:
            r = postprocess_layer(F.relu(r))
        return r


class Phi(nn.Module):
    def __init__(self, params, input_dim, output_dim, device="cpu"):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, params["phi_feature_dim"] * 2, (1, 1)).to(device)
        # self.conv2 = nn.Conv3d(params["phi_feature_dim"] * 4, params["phi_feature_dim"] * 4, (1, 1, 1)).to(device)
        self.conv2 = nn.Conv2d(params["phi_feature_dim"] * 2, output_dim, (1, 1)).to(device)
        self.ex = (torch.ones((params["num_users"], params["num_users"])) -
                   torch.eye(params["num_users"])).requires_grad_().to(device)
        self.n_cat1 = params["num_users"] - 1
        self.feature_dim = params["phi_feature_dim"]
        self.num_users = params["num_users"]

    def forward(self, phi_input):
        def postprocess_layer(conv_output):
            features_c0 = conv_output[:, : self.feature_dim, :, :]
            features_c1 = self.ex @ conv_output[:, self.feature_dim: (self.feature_dim * 2), :, :] / self.n_cat1
            return torch.cat((features_c0, features_c1), 1)

        r = F.relu(self.conv1(phi_input))
        r = postprocess_layer(r)
        # r = F.relu(self.conv2(r))
        # r = postprocess_layer(r)
        r = self.conv2(r)
        r = postprocess_layer(r)
        return r


class GNN(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.params = params
        self.device = device
        phi_output_dim = params["phi_feature_dim"] * 2
        gamma_output_dim = params["gamma_feature_dim"] * 2
        if params["reduced_msg_input"]:
            phi_input_dim = gamma_output_dim * 1 + params["input_feature_dim"]
            phi_input_dim_init = params["input_feature_dim"] * 2
        else:
            phi_input_dim = gamma_output_dim * 2 + params["input_feature_dim"]
            phi_input_dim_init = 3 * params["input_feature_dim"]
        gamma_input_dim_init = params["input_feature_dim"] + phi_output_dim
        gamma_input_dim = gamma_output_dim + phi_output_dim

        self.phi1 = Phi(params, phi_input_dim_init, phi_output_dim, device)
        self.phi2 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi3 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi4 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi5 = Phi(params, phi_input_dim, phi_output_dim, device)

        self.gamma1 = Gamma(params, gamma_input_dim_init, gamma_output_dim, device)
        self.gamma2 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma3 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma4 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma5 = Gamma(params, gamma_input_dim, self.params["max_conns_ap"], device, final_layer=True)

        self.ex = (torch.ones((params["num_aps"], params["num_aps"])) -
                   torch.eye(params["num_aps"])).requires_grad_().to(device)

    def forward(self, channel_gains):
        def process_edge(edge_features, node_features, phi):
            phi_input = torch.cat([edge_features, node_features], dim=1)
            phi_output = phi(phi_input)
            return phi_output

        def process_node(msgs, node_features, gamma):
            agg_msgs = msgs @ self.ex / (self.params["num_aps"] - 1)
            gamma_input = torch.cat([node_features, agg_msgs], dim=1)
            gamma_output = gamma(gamma_input)
            return gamma_output

        msgs = process_edge(channel_gains, channel_gains, self.phi1)
        r = process_node(msgs, channel_gains, self.gamma1)

        msgs = process_edge(channel_gains, r, self.phi2)
        r = process_node(msgs, r, self.gamma2)

        msgs = process_edge(channel_gains, r, self.phi3)
        r = process_node(msgs, r, self.gamma3)

        msgs = process_edge(channel_gains, r, self.phi4)
        r = process_node(msgs, r, self.gamma4)

        msgs = process_edge(channel_gains, r, self.phi5)
        r = process_node(msgs, r, self.gamma5)

        return r


class GNNRefinement(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.params = params
        self.device = device
        phi_output_dim = params["phi_feature_dim"] * 2
        gamma_output_dim = params["gamma_feature_dim"] * 2
        input_feature_dim = params["input_feature_dim"] + 1 + params["max_conns_ap"]
        if params["reduced_msg_input"]:
            phi_input_dim = gamma_output_dim * 1 + input_feature_dim
            phi_input_dim_init = 2 * input_feature_dim
        else:
            phi_input_dim = gamma_output_dim * 2 + input_feature_dim
            phi_input_dim_init = 3 * input_feature_dim
        gamma_input_dim_init = input_feature_dim + phi_output_dim
        gamma_input_dim = gamma_output_dim + phi_output_dim

        self.phi1 = Phi(params, phi_input_dim_init, phi_output_dim, device)
        self.phi2 = Phi(params, phi_input_dim, phi_output_dim, device)

        self.gamma1 = Gamma(params, gamma_input_dim_init, gamma_output_dim, device)
        self.gamma2 = Gamma(params, gamma_input_dim, self.params["max_conns_ap"], device, final_layer=True)

        self.ex = (torch.ones((params["num_aps"], params["num_aps"])) -
                   torch.eye(params["num_aps"])).requires_grad_().to(device)

    def forward(self, channel_gains):
        def process_edge(edge_features, node_features, phi):
            phi_input = torch.cat([edge_features, node_features], dim=1)
            phi_output = phi(phi_input)
            return phi_output

        def process_node(msgs, node_features, gamma):
            agg_msgs = msgs @ self.ex / (self.params["num_aps"] - 1)
            gamma_input = torch.cat([node_features, agg_msgs], dim=1)
            gamma_output = gamma(gamma_input)
            return gamma_output

        msgs = process_edge(channel_gains, channel_gains, self.phi1)
        r = process_node(msgs, channel_gains, self.gamma1)

        msgs = process_edge(channel_gains, r, self.phi2)
        r = process_node(msgs, r, self.gamma2)

        return torch.nn.functional.softplus(r) / 10


class GNNDeterministicGumbel(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.params = params
        self.device = device
        phi_output_dim = params["phi_feature_dim"] * 2
        gamma_output_dim = params["gamma_feature_dim"] * 2
        if params["reduced_msg_input"]:
            phi_input_dim = gamma_output_dim * 1 + params["input_feature_dim"]
            phi_input_dim_init = params["input_feature_dim"] * 2
        else:
            phi_input_dim = gamma_output_dim * 2 + params["input_feature_dim"]
            phi_input_dim_init = 3 * params["input_feature_dim"]
        gamma_input_dim_init = params["input_feature_dim"] + phi_output_dim
        gamma_input_dim = gamma_output_dim + phi_output_dim

        self.phi1 = Phi(params, phi_input_dim_init, phi_output_dim, device)
        self.phi2 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi3 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi4 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi5 = Phi(params, phi_input_dim, phi_output_dim, device)

        self.gamma1 = Gamma(params, gamma_input_dim_init, gamma_output_dim, device)
        self.gamma2 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma3 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma4 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma5 = Gamma(params, gamma_input_dim, self.params["max_conns_ap"], device, final_layer=True)

        self.ex = (torch.ones((params["num_aps"], params["num_aps"])) -
                   torch.eye(params["num_aps"])).requires_grad_().to(device)

    def forward(self, channel_gains, temp):
        def process_edge(edge_features, node_features, phi):
            phi_input = torch.cat([edge_features, node_features], dim=1)
            phi_output = phi(phi_input)
            return phi_output

        def process_node(msgs, node_features, gamma):
            agg_msgs = msgs @ self.ex / (self.params["num_aps"] - 1)
            gamma_input = torch.cat([node_features, agg_msgs], dim=1)
            gamma_output = gamma(gamma_input)
            return gamma_output

        msgs = process_edge(channel_gains, channel_gains, self.phi1)
        r = process_node(msgs, channel_gains, self.gamma1)

        msgs = process_edge(channel_gains, r, self.phi2)
        r = process_node(msgs, r, self.gamma2)

        msgs = process_edge(channel_gains, r, self.phi3)
        r = process_node(msgs, r, self.gamma3)

        msgs = process_edge(channel_gains, r, self.phi4)
        r = process_node(msgs, r, self.gamma4)

        msgs = process_edge(channel_gains, r, self.phi5)
        r = process_node(msgs, r, self.gamma5)

        assignment = gumbel_softmax(y=r, tau=temp, dim=-2)
        return r, assignment


class GNNDeterministicRecursiveGumbel(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.params = params
        self.device = device
        phi_output_dim = params["phi_feature_dim"] * 2
        gamma_output_dim = params["gamma_feature_dim"] * 2
        if params["reduced_msg_input"]:
            phi_input_dim = gamma_output_dim * 1 + params["input_feature_dim"]
            phi_input_dim_init = params["input_feature_dim"] * 2
        else:
            phi_input_dim = gamma_output_dim * 2 + params["input_feature_dim"]
            phi_input_dim_init = 3 * params["input_feature_dim"]
        gamma_input_dim_init = params["input_feature_dim"] + phi_output_dim
        gamma_input_dim = gamma_output_dim + phi_output_dim

        self.phi1 = Phi(params, phi_input_dim_init, phi_output_dim, device)
        self.phi2 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi3 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi4 = Phi(params, phi_input_dim, phi_output_dim, device)
        self.phi5 = Phi(params, phi_input_dim, phi_output_dim, device)

        self.gamma1 = Gamma(params, gamma_input_dim_init, gamma_output_dim, device)
        self.gamma2 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma3 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma4 = Gamma(params, gamma_input_dim, gamma_output_dim, device)
        self.gamma5 = Gamma(params, gamma_input_dim, self.params["max_conns_ap"], device, final_layer=True)

        self.gamma_r = Gamma(params, self.params["max_conns_ap"] + 1, self.params["max_conns_ap"], device, final_layer=True)

        self.ex = (torch.ones((params["num_aps"], params["num_aps"])) -
                   torch.eye(params["num_aps"])).requires_grad_().to(device)

    def forward(self, channel_gains, temp):
        def process_edge(edge_features, node_features, phi):
            phi_input = torch.cat([edge_features, node_features], dim=1)
            phi_output = phi(phi_input)
            return phi_output

        def process_node(msgs, node_features, gamma):
            agg_msgs = msgs @ self.ex / (self.params["num_aps"] - 1)
            gamma_input = torch.cat([node_features, agg_msgs], dim=1)
            gamma_output = gamma(gamma_input)
            return gamma_output

        msgs = process_edge(channel_gains, channel_gains, self.phi1)
        r = process_node(msgs, channel_gains, self.gamma1)

        msgs = process_edge(channel_gains, r, self.phi2)
        r = process_node(msgs, r, self.gamma2)

        msgs = process_edge(channel_gains, r, self.phi3)
        r = process_node(msgs, r, self.gamma3)

        msgs = process_edge(channel_gains, r, self.phi4)
        r = process_node(msgs, r, self.gamma4)

        msgs = process_edge(channel_gains, r, self.phi5)
        r = process_node(msgs, r, self.gamma5)

        assignment = gumbel_softmax(y=r, tau=temp, dim=-2)
        return r, assignment


def gumbel_softmax(y, tau, dim):
    y = y - y.max(dim, keepdim=True)[0]
    while torch.any(torch.isinf(y / tau)):
        tau *= 1.1
    y = torch.exp(y / torch.clamp(tau, min=1e-3))
    y = y / y.sum(dim, keepdim=True)
    return y


class GNNGumbel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.gnn_a = GNN(params, device)
        self.gnn_l = GNN(params, device)
        self.softplus = nn.Softplus()

        # self.temp1 = nn.Linear(self.gnn_a.params["max_conns_ap"] * 2, 10).to(device)
        # self.temp2 = nn.Linear(10, 10).to(device)
        # self.temp3 = nn.Linear(10, 1).to(device)

    def forward(self, gnn_input):
        a = self.gnn_a(gnn_input)
        l = self.softplus(self.gnn_l(gnn_input) + 4) + 1e-4
        b = a + l
        dis = Uniform(a, b)
        p = dis.rsample()

        support = l.mean(dim=[1, 2, 3])
        return p, support



def sumrate(channel, assignment, params):
    rss = (channel[:, 0, :, :] * torch.max(assignment, dim=1).values).sum(dim=2)
    sum_rate = torch.log2(1 + (rss * 1e-3 / params["noise_power"])).sum(dim=1)
    return sum_rate


# Every user is connected to at least min_conns_ue APs
def compute_conn_deficiency(assignment, params, averaging_users=True):
    n = torch.relu(params["min_conns_ue"] - torch.sum(torch.max(assignment, dim=1).values, dim=2))
    if averaging_users:
        return torch.mean(n, dim=1)
    else:
        return n


def calc_conn_penalty(p: torch.tensor, params, device):
    temp = torch.tensor(1e-3).to(device)
    assignment = gumbel_softmax(y=p, tau=temp, dim=-2)
    deficiency = compute_conn_deficiency(assignment, params, False).detach()
    deficiency = torch.round(deficiency)
    diff = p.max(dim=2, keepdim=True).values - p
    penalty = diff * deficiency[:, None, :, None]
    penalty = torch.relu((penalty / 5 + 1e-8) ** 0.3 + penalty - 1e-8 ** 0.3)

    # Debug
    # pp = torch.clone(p).detach().requires_grad_()
    # opt = Adam([pp], lr=1e-4)
    # for i in range(10000):
    #     opt.zero_grad()
    #     assignment = gumbel_softmax(y=pp, tau=temp, dim=-2)
    #     deficiency = compute_conn_deficiency(assignment, params, False).detach()
    #     deficiency = torch.round(deficiency)
    #     diff = pp.max(dim=2, keepdim=True).values - pp
    #     penalty = diff * deficiency[:, None, :, None]
    #     penalty = torch.relu((penalty / 1 + 1e-8) ** 0.3 + penalty - 1e-8 ** 0.3)
    #     penalty.sum(dim=[1, 2, 3]).mean().backward()
    #     opt.step()
    #     pass

    return torch.sum(penalty, dim=[1, 2, 3])


def productrate(channel, power, params):
    power = power[:, :, :, None, :]
    rss = (channel * power).sum(dim=[1, 2])
    product_rate = 0
    for user_idx in range(params["num_users"]):
        product_rate = product_rate + torch.log(torch.log2(1 + rss[:, user_idx, user_idx] /
                                                   (torch.sum(rss[:, user_idx, :], dim=1) - rss[:, user_idx, user_idx] +
                                                    params["noise_power"])))
    return product_rate


def determine_eta(temperature, previous_kappa):
    lr_kappa = 100
    if len(temperature) < 100:
        return previous_kappa
    elif temperature[-1] >= np.mean(temperature):
        return previous_kappa + lr_kappa
    else:
        return F.relu(previous_kappa - lr_kappa / 2)


def determine_kappa(support, previous_kappa):
    lr_kappa = 1e-2
    if len(support) < 30:
        return previous_kappa
    elif support[-1] >= np.mean(support):
        return previous_kappa + lr_kappa
    else:
        return F.relu(previous_kappa - lr_kappa / 2)


def determine_penalty_factor(ref, previous_factor, lr, patient):
    if len(ref) < patient:
        return previous_factor
    elif ref[-1] >= np.mean(ref):
        return previous_factor + lr
    else:
        return F.relu(previous_factor - lr / 2)


def determine_temperature(counter, total_epoches):
    return torch.tensor(1e-2 ** (counter / total_epoches))
    # return torch.tensor((1 - counter / total_epoches))


def alm_update_lambda_mu(old_lambda, old_mu, delta_mu, maximum, residual):
    mu = torch.tensor((old_mu + delta_mu).item())
    lambdaa = old_lambda + old_mu * residual.mean().item()
    return torch.minimum(lambdaa, torch.tensor(maximum)), torch.minimum(mu, torch.tensor(maximum))


def whether_converged(indicator, obj="min", patience=100):
    if len(indicator) < patience:
        return False
    else:
        mean = np.mean(indicator)
        # std = np.std(indicator)
        if obj == "min":
            return indicator[-1] >= mean
        else:
            return indicator[-1] <= mean


def dc3(assignment, deficiency):
    ...


class CFData(Dataset):
    def __init__(self, params, path="", device="cpu", test=False):
        super().__init__()
        self.params = params
        self.test = test
        self.path = path
        self.device = device

        f = np.load(self.path)
        self.channels = torch.tensor(f.f.channels, dtype=torch.float32, device=device)
        self.channels = self.channels[:, None, :, :]
        self.ue_pos = torch.tensor(f.f.ue_pos, dtype=torch.float32, device=device)

    def one_hot(self):
        max_indices = torch.argmax(self.channels, dim=-2)
        one_hot_tensor = torch.zeros_like(self.channels)
        one_hot_tensor.scatter_(2, max_indices.unsqueeze(2), 1)
        self.channels *= one_hot_tensor
        self.channels.detach()

    def __getitem__(self, item):
        return item, self.channels[item, :, :], self.ue_pos[item, :, :]

    def __len__(self):
        return self.channels.shape[0]

