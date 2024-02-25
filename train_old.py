from corev2 import CFData, sumrate, alm_update_lambda_mu, compute_conn_deficiency, whether_converged
from corev2 import determine_temperature, calc_conn_penalty, gumbel_softmax
from corev2 import GNNGumbel, GNNRefinement
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import argparse
import datetime
from pathlib import Path
from params import params
from collections import deque

torch.set_default_dtype(torch.float32)
record = False
tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False
    record = False


def prepare_data(params, path: str, mode: str):
    params["data_available"] = True
    dataset_path = path.split("/")
    params["num_aps"], params["num_users"] = [int(i) for i in dataset_path[-1].split("_") if i.isdigit()]
    if params["preselection_ap"] >= params["num_users"]:
        params["preselection_ap"] = int(params["num_users"] / params["num_aps"]) + 5
    else:
        params["preselection_ap"] = 15

    if mode.lower() == "train":
        params["channels_path"] = f'{path}/channels_training_{params["num_aps"]}_aps_{params["num_users"]}_users'
        params[
            "required_rates_path"] = f'{path}/required_rates_training_{params["num_aps"]}_aps_{params["num_users"]}_users.pt'
        params["postions"] = f'{path}/positions_{params["num_aps"]}_aps_{params["num_users"]}_users.pt'

    elif mode.lower() == "test":
        params["channels_path"] = f'{path}/channels_testing_{params["num_aps"]}_aps_{params["num_users"]}_users'
        params[
            "required_rates_path"] = f'{path}/required_rates_testing_{params["num_aps"]}_aps_{params["num_users"]}_users.pt'
        params["postions"] = f'{path}/positions_{params["num_aps"]}_aps_{params["num_users"]}_users.pt'

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--record")
    parser.add_argument("--num_data_samples", type=int)
    # if data is present then compute the number of chunks based on the available data else use the number of chunks provided
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dataset_path")
    args = parser.parse_args()

    path = f'data/{params["num_users"]}ue_{params["num_aps"]}aps'  # initialize the path variable to a default path

    # in order to record the training behaviour with tensorboard
    if args.record is not None:
        record = tb and args.record == "True"

    # torch.set_default_dtype(torch.float32)

    # selects the device to run the machine learning algorithm from
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # record training data
    if record:
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        Path(params["results_path"] + dt_string).mkdir(parents=True, exist_ok=True)
        params["results_path"] = params["results_path"] + dt_string + "/"

    # if number of samples is passed as an argument, use that data instead of the default one present in the params file
    if args.num_data_samples is not None:
        params["num_data_samples"] = int(args.num_data_samples)

    # if the dataset path is passed as an argument, load data from the specified path
    if args.dataset_path is not None:
        path = args.dataset_path
        params["data_available"] = True
        chunks = len(os.listdir(path))
        if chunks == 0:
            print(f'data not available at the specified directory {path}')
        else:
            params["num_samples_chunks"] = np.ceil(params["num_data_samples"] / chunks)

    model = GNNGumbel(params, device)
    refinement_model = GNNRefinement(params, device)
    # model = torch.compile(GNNGumble(params, device))

    # Debug
    # model.load_state_dict(torch.load("results/21-10-2023_22-24-24/model_160000", map_location=torch.device("cpu")))

    dataset = CFData(params, path=path + "/training_data.npz", device=device)
    dataset_testing = CFData(params, path=path + "/testing_data.npz", device=device, test=True)
    train_loader = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), params["lr"] / params["gradient_accumulation"])
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  mode="min",
                                  factor=0.9,
                                  patience=2000,
                                  cooldown=4000,
                                  min_lr=params["min_lr"])

    model.train()
    refinement_model.train()
    counter = 1

    lambda_conn_deficiency = torch.ones(dataset.__len__()).to(device) * 10
    # lambda_temp = torch.ones(dataset.__len__()).to(device) * 10
    lambda_support = torch.zeros(dataset.__len__()).to(device)

    support_history = list()
    for _ in range(len(dataset)):
        support_history.append(deque())

    # temp_history = list()
    # for _ in range(len(dataset)):
    #     temp_history.append(deque())

    conn_deficiency_history = list()
    for _ in range(len(dataset)):
        conn_deficiency_history.append(deque())

    print('training started..')
    if record:
        writer = SummaryWriter(logdir=params["results_path"])
    optimizer.zero_grad()
    while True:
        for indices, channels, ue_pos in train_loader:
            temp = determine_temperature(counter, params["epoch"])
            p, support = model((channels - params["mean_channel"]) / params["std_channel"])
            conn_penalty = calc_conn_penalty(p, params, device)
            lambda_support_pt = lambda_support[indices]
            lambda_conn_deficiency_pt = lambda_conn_deficiency[indices]
            loss = conn_penalty
            for refinement_idx in range(5):
                hard_assignment = gumbel_softmax(p, torch.tensor(1e-4).to(device), -2)
                deficiency = torch.round(compute_conn_deficiency(hard_assignment, params, False))
                conn_penalty = calc_conn_penalty(p, params, device)
                loss += lambda_conn_deficiency_pt * conn_penalty
                deficiency = deficiency[:, None, :, None].repeat((1, 1, 1, params["num_aps"]))
                refinement_input = torch.cat([deficiency, p,
                                              (channels - params["mean_channel"]) / params["std_channel"]], dim=1)
                delta_p = refinement_model(refinement_input)
                p += delta_p * deficiency
            assignment = gumbel_softmax(p, temp, -2)
            sr = sumrate(10 ** (channels / 10), assignment, params)
            loss = (loss + lambda_support_pt * support).mean()

            loss.backward()
            if counter % params["gradient_accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
                optimizer.step()
                optimizer.zero_grad()

            # Perform the augmented Lagrangian step
            for index_in_this_batch, channel_idx in enumerate(indices):
                while len(support_history[channel_idx]) > params["patience"]:
                    support_history[channel_idx].popleft()
                while len(conn_deficiency_history[channel_idx]) > params["patience"]:
                    conn_deficiency_history[channel_idx].popleft()
                support_history[channel_idx].append(support[index_in_this_batch].item())
                conn_deficiency_history[channel_idx].append(conn_penalty[index_in_this_batch].item())

                converged = (whether_converged(support_history[channel_idx], "min", params["patience"]) and
                             whether_converged(conn_deficiency_history[channel_idx], "min", params["patience"]))
                if converged:
                    lambda_conn_deficiency[channel_idx] = alm_update_lambda_mu(lambda_conn_deficiency[channel_idx],
                                                                               1,
                                                                               conn_penalty[index_in_this_batch])
                    lambda_support[channel_idx] = alm_update_lambda_mu(lambda_support[channel_idx],
                                                                       1,
                                                                       support[index_in_this_batch])

            if counter > 5e4:
                scheduler.step(loss)

            with torch.no_grad():
                channels = dataset_testing.channels
                # sparse_channels = dataset_testing.sparse_channels
                p_test, support_test = model((channels - params["mean_channel"]) / params["std_channel"])
                for refinement_idx in range(5):
                    hard_assignment_test = gumbel_softmax(p_test, torch.tensor(1e-4).to(device), -2)
                    deficiency_test = torch.round(compute_conn_deficiency(hard_assignment_test, params, False))
                    conn_penalty_test = calc_conn_penalty(p_test, params, device)
                    deficiency_test = deficiency_test[:, None, :, None].repeat((1, 1, 1, params["num_aps"]))
                    refinement_input_test = torch.cat([deficiency_test, p_test,
                                                  (channels - params["mean_channel"]) / params["std_channel"]], dim=1)
                    delta_p_test = refinement_model(refinement_input_test)
                    p_test += delta_p_test * deficiency_test
                assignment_test = gumbel_softmax(p_test, temp, -2)
                sr_test = sumrate(10 ** (channels / 10), assignment_test, params)

            print(
                f"Iter={counter}, loss={loss.mean()}, rate={sr.mean()}, temperature={temp.mean()}, deficiency={conn_penalty.mean()}, support={support.mean()}")

            if record and counter % 100 == 0:
                writer.add_scalar("Training/sum_rate", sr.mean().item(), counter)
                writer.add_scalar("Training/temperature", temp.mean().item(), counter)
                writer.add_scalar("Training/conn_penalty", conn_penalty.mean().item(), counter)
                writer.add_scalar("Training/loss", loss.mean().item(), counter)
                writer.add_scalar("Training/lambda_support", lambda_support.mean().item(), counter)
                writer.add_scalar("Training/lambda_conn_deficiency", lambda_conn_deficiency.mean().item(), counter)
                writer.add_scalar("Training/support", support.mean().item(), counter)
                writer.add_scalar("Testing/support", support_test.mean().item(), counter)
                writer.add_scalar("Testing/sum_rate", sr_test.mean().item(), counter)
                writer.add_scalar("Testing/conn_deficiency", conn_penalty_test.mean().item(), counter)
                writer.add_scalar("Training/lr", optimizer.param_groups[0]["lr"], counter)
                torch.save(model.state_dict(), params["results_path"] + "model_{iter}".format(iter=counter))
                torch.save(refinement_model.state_dict(), params["results_path"] + "r_model_{iter}".format(iter=counter))
            counter += 1

        if counter >= params["epoch"]:
            break
