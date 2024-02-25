import copy

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from torch.distributions import Uniform


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
        self.gamma5 = Gamma(params, gamma_input_dim, 1, device, final_layer=True)

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


class GNNGumbelRecursive(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.gnn = GNN(params, device)
        self.softplus = nn.Softplus()

    def forward(self, gnn_input):
        deficiency = torch.ones_like(gnn_input)[:, :1, :, :1] * self.gnn.params["min_conns_ue"]
        # assignment_list = list()
        support_list = list()
        total_assignment = torch.zeros_like(gnn_input)[:, :1, :, :]
        assignment = torch.zeros_like(gnn_input)[:, :1, :, :]
        for iter in range(self.gnn.params["max_conns_ap"]):
            gnn_input_with_deficiency = torch.cat((gnn_input,
                                                   assignment,
                                                   deficiency.repeat(1, 1, 1, self.gnn.params["num_aps"])), dim=1)
            a = self.gnn(gnn_input_with_deficiency)
            #a = a - torch.mean(a, dim=-2, keepdim=True)
            a=50*torch.tanh(2*torch.tanh(a))
            # l = self.softplus(self.gnn_l(gnn_input_with_deficiency) + 4) + 1e-4
            # b = a + l
            if torch.any(torch.isnan(a)):
                print("NaN as output :-(")
                exit()
            # if torch.any(a >= b):
            #     print("Unexpected: a is greater than or equal to b in some elements")
            #     print("a values:", a[a >= b])
            #     print("b values:", b[a >= b])
            #     exit()
            # dis = Uniform(a, b)
            p = a
            assignment = torch.nn.functional.softmax(p, dim=-2)
            old_total_assignment = total_assignment
            total_assignment = torch.max(assignment, old_total_assignment)
            diff_assignment = total_assignment - old_total_assignment
            # assignment_list.append(assignment)
            deficiency -= diff_assignment.sum(dim=-1, keepdim=True)
            pass
        # total_assignment.retain_grad()

        return (total_assignment,
                torch.sum(torch.relu(deficiency), dim=[1, 2, 3]))



def calc_discreteness_penalty(p, sum_up=True):#0 这里加了1e-10
    if sum_up:
        return torch.sum(torch.sqrt(p - p ** 2+ 1e-10), dim=[1, 2, 3])
    else:
        return torch.sqrt(p - p ** 2)


def calc_discreteness_penalty2(p, sum_up=True):
    if sum_up:
        return -torch.sum(torch.log(p + 1e-6) * p, dim=[1, 2, 3])
    else:
        return -torch.sum(torch.log(p + 1e-6) * p, dim=-2)

def calc_alm_penalty(penalty, mu, lambdaa):
    # return torch.relu((penalty / 5) ** 0.3 + penalty)
    return lambdaa * penalty + mu / 2 * penalty ** 2
