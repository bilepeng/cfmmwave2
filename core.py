import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.utils.data import Dataset
import os


class Phi(nn.Module):
    def __init__(self, params, input_dim, output_dim, device="cpu"):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, params["phi_feature_dim"] * 4, (1, 1)).to(device)
        self.conv2 = nn.Conv2d(params["phi_feature_dim"] * 4, params["phi_feature_dim"] * 4, (1, 1)).to(device)
        self.conv3 = nn.Conv2d(params["phi_feature_dim"] * 4, output_dim, (1, 1)).to(device)
        self.ex = (torch.ones((params["preselection_ap"], params["preselection_ap"])) -
                   torch.eye(params["preselection_ap"])).requires_grad_().to(device)
        self.n_cat1 = params["num_users"] - 1
        self.n_cat2 = params["num_users"] - 1
        self.n_cat3 = (params["num_users"] - 1) ** 2
        self.feature_dim = params["phi_feature_dim"]
        self.num_users = params["num_users"]

    def forward(self, phi_input):
        def postprocess_layer(conv_output):
            features_c0 = conv_output[:, : self.feature_dim, :, :]
            features_c1 = self.ex @ conv_output[:, self.feature_dim: (self.feature_dim * 2), :, :] / self.n_cat1
            features_c2 = conv_output[:, (self.feature_dim * 2): (self.feature_dim * 3), :, :] @ self.ex / self.n_cat2
            features_c3 = (self.ex @ conv_output[:, (self.feature_dim * 3): (self.feature_dim * 4), :, :]
                           @ self.ex / self.n_cat3)
            return torch.cat((features_c0, features_c1, features_c2, features_c3), 1)

        r = F.relu(self.conv1(phi_input))
        r = postprocess_layer(r)
        # r = F.relu(self.conv2(r))
        # r = postprocess_layer(r)
        r = F.relu(self.conv3(r))
        r = postprocess_layer(r)
        return r


class Gamma(nn.Module):
    def __init__(self, params, input_dim, output_dim, device="cpu", final_layer=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, params["gamma_feature_dim"] * 4, (1, 1)).to(device)
        self.conv2 = nn.Conv2d(params["gamma_feature_dim"] * 4, params["gamma_feature_dim"] * 4, (1, 1)).to(device)
        self.conv3 = nn.Conv2d(params["gamma_feature_dim"] * 4, output_dim, (1, 1)).to(device)
        self.ex = (torch.ones((params["preselection_ap"], params["preselection_ap"])) -
                   torch.eye(params["preselection_ap"])).requires_grad_().to(device)
        self.n_cat1 = params["num_users"] - 1
        self.n_cat2 = params["num_users"] - 1
        self.n_cat3 = (params["num_users"] - 1) ** 2
        self.feature_dim = params["gamma_feature_dim"]
        self.num_users = params["num_users"]
        self.final_layer = final_layer

    def forward(self, gamma_input):
        def postprocess_layer(conv_output):
            features_c0 = conv_output[:, : self.feature_dim, :, :]
            features_c1 = self.ex @ conv_output[:, self.feature_dim: (self.feature_dim * 2), :, :] / self.n_cat1
            features_c2 = conv_output[:, (self.feature_dim * 2): (self.feature_dim * 3), :, :] @ self.ex / self.n_cat2
            features_c3 = (self.ex @ conv_output[:, (self.feature_dim * 3): (self.feature_dim * 4), :, :]
                           @ self.ex / self.n_cat3)
            return torch.cat((features_c0, features_c1, features_c2, features_c3), 1)

        r = F.relu(self.conv1(gamma_input))
        r = postprocess_layer(r)
        # r = F.relu(self.conv2(r))
        # r = postprocess_layer(r)
        r = self.conv3(r)
        if not self.final_layer:
            r = postprocess_layer(F.relu(r))
        return r


class GNN(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.params = params
        self.device = device
        phi_output_dim = params["phi_feature_dim"] * 4
        gamma_output_dim = params["gamma_feature_dim"] * 4
        if params["reduced_msg_input"]:
            phi_input_dim = gamma_output_dim * 1 + params["input_feature_dim"]
            phi_input_dim_init = 2 * params["input_feature_dim"]
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

        self.gamma_power_portion1 = nn.Conv2d(gamma_output_dim, gamma_output_dim, (1, 1)).to(device)
        self.gamma_power_portion2 = nn.Conv2d(gamma_output_dim, gamma_output_dim, (1, 1)).to(device)
        self.gamma_power_portion3 = nn.Conv2d(gamma_output_dim, 1, (1, 1)).to(device)
        self.sigmoid = nn.Sigmoid()
        self.gamma_total_power1 = nn.Conv2d(gamma_output_dim, gamma_output_dim, (1, 1)).to(device)
        self.gamma_total_power2 = nn.Conv2d(gamma_output_dim, 1, (1, 1)).to(device)

    def forward(self, gnn_input, selection):
        def edge_processing(node_features, edge_features, phi):
            if self.params["reduced_msg_input"]:
                raw_edge_output = list()
                for x_j, e_ji, s in zip(node_features, edge_features, selection):
                    phi_input = torch.cat((x_j, e_ji), dim=1)
                    raw_edge_output.append(s @ phi(phi_input) @ s.transpose(2, 3))
                edge_output = list()
                for i in range(self.params["num_aps"]):
                    edge_output.append([selection[i].transpose(2, 3) @ o @ selection[i]
                                        for j, o in enumerate(raw_edge_output) if j != i])
            else:
                edge_output = list()
                for i, x_i in enumerate(node_features):
                    output4i = list()
                    phi_input = torch.cat((x_j, x_i, e_ji), dim=1)
                    for j, x_j, e_ji in zip(range(self.params["num_aps"]), node_features, edge_features):
                        if i != j:
                            output4i.append(phi(phi_input))
                    edge_output.append(output4i)
            return edge_output

        def node_processing(node_features, edge_output, gamma):
            node_output = list()
            for x_i, msgs in zip(node_features, edge_output):
                gamma_input = torch.cat((x_i, torch.stack(msgs, dim=0).mean(dim=0)), dim=1)
                node_output.append(gamma(gamma_input))
            return node_output

        r_edges = edge_processing(gnn_input, gnn_input, self.phi1)
        r_nodes = node_processing(gnn_input, r_edges, self.gamma1)

        r_edges = edge_processing(r_nodes, gnn_input, self.phi2)
        r_nodes = node_processing(r_nodes, r_edges, self.gamma2)

        r_edges = edge_processing(r_nodes, gnn_input, self.phi3)
        r_nodes = node_processing(r_nodes, r_edges, self.gamma3)

        r_edges = edge_processing(r_nodes, gnn_input, self.phi4)
        r_nodes = node_processing(r_nodes, r_edges, self.gamma4)

        r_edges = edge_processing(r_nodes, gnn_input, self.phi5)
        r_nodes = node_processing(r_nodes, r_edges, self.gamma5)

        gnn_output = list()
        for x_i in r_nodes:
            # gnn_output.append(torch.diagonal(self.gamma5(x_i), dim1=2, dim2=3))
            gnn_output.append(torch.diagonal(x_i, dim1=2, dim2=3))

        # node_input = list()
        # for r, g in zip(r_nodes, gnn_input):
        #     node_input.append(torch.cat((g, r), dim=1))
        # r_nodes = node_processing(node_input, r_edges, self.gamma2)

        # r_edges = edge_processing(r_nodes, gnn_input, self.phi3)
        # node_input.clear()
        # for r, g in zip(r_nodes, gnn_input):
        #     node_input.append(torch.cat((g, r), dim=1))
        # r_nodes = node_processing(node_input, r_edges, self.gamma3)
        # gnn_output = list()
        # for r in node_output:
        #     transmit_power = torch.diagonal(self.gamma_power_portion2(r), dim1=2, dim2=3)
        #     gnn_output.append(transmit_power)
        return gnn_output


class GNNGlobal(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.gnn_mu = GNN(params, device)
        self.gnn_sigma = GNN(params, device)

    def forward(self, gnn_input, selection):
        mu = self.gnn_mu(gnn_input, selection)
        sigma = [(s + 1) * self.gnn_mu.params["pmax"] for s in self.gnn_sigma(gnn_input, selection)]

        dis = [Normal(m, s) for m, s in zip(mu, sigma)]
        p = [torch.sigmoid(d.rsample()) * self.gnn_mu.params["pmax"] for d in dis]
        normalized_p = list()
        for pp in p:
            sum_power = torch.maximum(pp.sum(dim=[1, 2], keepdim=True), torch.tensor(self.gnn_mu.params["pmax"]))
            normalized_p.append(pp / sum_power)

        entropy = torch.sum(torch.stack([d.entropy() for d in dis], dim=0), dim=[0, 2, 3])
        return p, mu, sigma, entropy


class CFData(Dataset):
    def __init__(self, params, path="", device="cpu"):
        super().__init__()
        self.params = params
        self.path = path
        self.device = device
        self.samples_per_chunk = self.params["num_samples_chunks"]
        self.total_chunks = int(np.ceil(self.params["num_data_samples"] / self.samples_per_chunk))
        self.chunk_num = 0
        self.counter = 0
        self.channels, self.rate_requirements, self.selection_matrices = self.load_chunk(self.chunk_num, self.device,
                                                                                         self.path)

    def __getitem__(self, item):
        item = int(item % self.samples_per_chunk)
        self.counter += 1
        if self.counter > self.samples_per_chunk:
            self.chunk_num = (self.chunk_num + 1) % self.total_chunks  # iterate over the number of chunks
            self.channels, self.rate_requirements, self.selection_matrices = self.load_chunk(self.chunk_num,
                                                                                             self.device,
                                                                                             path=self.path)

        return (item, [d[item, :, :, :] for d in self.channels],
                0,
                [d[item, :, :] for d in self.selection_matrices])

    def __len__(self):
        return self.params["num_data_samples"]

    def preselect(self, channel):
        signal_channel = torch.diag(channel)
        _, indices = torch.topk(signal_channel, self.params["preselection_ap"])
        selection_mat = np.zeros((self.params["num_users"], self.params["preselection_ap"]), dtype=np.float32)
        for i, index in enumerate(indices):
            selection_mat[index, i] = 1
        selection_mat = torch.tensor(selection_mat)

        return selection_mat

    def load_chunk(self, chunk_num, device, path=""):
        """ Loads a previously saved chunk to memory.

        Args:
            chunk_num (int): The chunk id (integer) to be restored.
            path (str, optional): Optional. The path, where the chunks are stored.

        Returns:
            List, Tensor, List: Channels as a list with L items and each item of the shape n x VirtualUsers x VirtualUsers, rate requirement per user, selection matrix per AP with same configuration as channels.
        """

        # -------------------- Initialize locals -------------------- #
        file_path = os.path.join(path, f'chunk_{chunk_num}.npz')

        # -------------------- load chunks to memory ---------------- #
        file = np.load(file_path)
        channels, rate_req, selection = file.f.channels, file.f.rr, file.f.sel

        channels = torch.from_numpy(channels).to(device).unbind()
        rate_req = torch.from_numpy(rate_req).to(device).unbind()
        selection = torch.from_numpy(selection).to(device).unbind()

        channels = [c[:, None, :, :] for c in channels]
        selection = [m[:, None, :, :] for m in selection]  # To enable broadcast
        self.counter = 0  # reset the counter for number of items sampled

        return channels, rate_req, selection


def determine_kappa(support, previous_kappa):
    lr_kappa = 1e-6
    if len(support) < 10:
        return previous_kappa
    elif support[-1] >= np.mean(support[:-1]):
        return previous_kappa + lr_kappa
    else:
        return F.relu(previous_kappa - lr_kappa / 2)


def sumrate(channel, power, params):
    rss = [c[:, 0, :, :] * p for c, p in zip(channel, power)]
    rss = torch.sum(torch.stack(rss, dim=0), dim=0)
    sum_rate = 0
    for user_idx in range(params["num_users"]):
        sum_rate = sum_rate + torch.log2(1 + rss[:, user_idx, user_idx] /
                                         (torch.sum(rss[:, user_idx, :], dim=1) - rss[:, user_idx, user_idx] +
                                          params["noise_power"]))
    return sum_rate


def aggregate_input(channels, required_rates, selection, params):
    channels = [(c - params["mean_channel"]) / params["std_channel"] for c in channels]
    if params["objective"] == "sumrate":
        channels = [s.transpose(dim0=2, dim1=3) @ c @ s for s, c in zip(selection, channels)]
        return channels
    elif params["objective"] == "power":
        required_rates = ((required_rates[:, None, :, None] - params["mean_rate_requirements"])
                          / params["std_rate_requirements"]).repeat((1, 1, 1, params["num_users"]))
        gnn_input = [torch.cat((c, required_rates), dim=1) for c in channels]
        gnn_input = [s.transpose(2, 3) @ g @ s for s, g in zip(selection, gnn_input)]
        return gnn_input


def power_st_rates(channels, power, selection, required_rates, params, device="cpu"):
    deficiency = compute_rate_deficiency(channels, power, selection, required_rates, params, device)
    power_consumption = torch.stack(power, dim=3).sum(dim=[1, 2, 3])
    return deficiency * 1000 + power_consumption * (deficiency == 0)


def compute_rate_deficiency(channels, power, selection, required_rates, params, device="cpu"):
    rss = [c[:, 0, :, :] * (p @ s[:, 0, :, :].transpose(1, 2)) for c, p, s in zip(channels, power, selection)]
    rss = torch.sum(torch.stack(rss, dim=0), dim=0)
    deficiency = 0
    for user_idx in range(params["num_users"]):
        rate = torch.log2(1 + rss[:, user_idx, user_idx] /
                          (torch.sum(rss[:, user_idx, :], dim=1) - rss[:, user_idx, user_idx] + params["noise_power"]))
        deficiency += torch.maximum(torch.zeros(1).to(device), required_rates[:, user_idx] - rate)
    return deficiency


def power_constraint(power, params):
    penalty = 0
    for p in power:
        penalty += torch.maximum(torch.zeros(1), p.sum(dim=[1, 2]) - params["pmax"])
    return penalty


