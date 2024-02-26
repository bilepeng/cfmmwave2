from corev2 import CFData, sumrate, alm_update_lambda_mu, compute_conn_deficiency, whether_converged
from corev2 import determine_temperature, calc_conn_penalty, gumbel_softmax
from corev3 import GNNGumbelRecursive as GNNGumbel
from corev3 import calc_discreteness_penalty, calc_alm_penalty, calc_discreteness_penalty2
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
import sys

torch.set_default_dtype(torch.float32)
record = True
tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False
    record = False

def record_w(writer, counter, **kwargs):
    for n, v in kwargs.items():
        if v is not None and hasattr(v, 'mean'):
            v = v.float()
            writer.add_scalar(n, v.mean().item(), counter)
def save_model(model, counter, params, optimizer, scheduler ,n_models= sys.maxsize):
    cp = {
        'epoch': counter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(cp,os.path.join(params["results_path"], f"model_{counter + 1}.pt"))

    all_models = [f for f in os.listdir(params["results_path"]) if
                  f.startswith("model_") and f.endswith(".pt")]
    s_models = sorted(all_models, key=lambda x: int(x.split("_")[1].split(".")[0]))
    while len(s_models) > n_models:
        os.remove(os.path.join(params["results_path"], s_models[0]))
        s_models.pop(0)
def testing(model, data, params, counter):
    with torch.no_grad():
        channels = data.channels
        p_test, deficiency_test = model((channels - params["mean_channel"]) / params["std_channel"])
        sr_test = sumrate(10 ** (channels / 10), p_test, params)
        #平均每张图有多少个user不符合预期
        _, k = torch.topk(p_test, k=params["max_conns_ap"], dim=2)
        compute_conn_deficiency = torch.sum(torch.relu(params["min_conns_ue"] - torch.sum(torch.zeros_like(p_test).scatter_(2, k, 1), dim=-1)),dim=-1)
        discreteness_penalty_test = None
        if counter > params["epoch"] * 2:
            discreteness_penalty_test = calc_discreteness_penalty2(p_test)
    return sr_test, deficiency_test, discreteness_penalty_test,compute_conn_deficiency


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
    elif torch.cuda.is_available():
        # device = torch.device("mps")
        device = "cuda"
    else:
        device = "cpu"

    # else:
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = "cpu"

    # record training data
    if record:
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        Path(params["results_path"] + dt_string).mkdir(parents=True, exist_ok=True)
        params["results_path"] = params["results_path"] + dt_string + "/"
        p_s = ["num_aps", "num_users", "mean_channel", "std_channel", "gradient_accumulation", "epoch",
               "batch_size"]
        with open(params["results_path"] + dt_string + ".txt", 'w') as f:
            for key in p_s:
                f.write(f"{key}: {params[key]}\n")

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

    # Debug
    #model.load_state_dict(torch.load("results/09-12-2023_10-10-05/model_100001.pt", map_location=torch.device("cpu")))
    dataset = CFData(params, path=path + "/training_data.npz", device=device)
    dataset_testing = CFData(params, path=path + "/testing_data.npz", device=device, test=True)
    train_loader = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), params["lr"] / params["gradient_accumulation"])
    scheduler = ReduceLROnPlateau(optimizer=optimizer,          #1 这里也修改了
                                  mode="min",
                                  factor=0.5,
                                  patience=params["epoch"]/5,
                                  cooldown=params["epoch"]/10,
                                  min_lr=params["min_lr"],
                                  verbose=True)
    #新的保存模型的函数里加入optimizer，scheduler和counter的存储。所以加载的时候
    # cp = torch.load("results/01-01-2024_23-45-17/model_300901.pt", map_location=torch.device("cpu"))
    # model.load_state_dict(cp['model_state_dict'])
    # optimizer.load_state_dict(cp['optimizer_state_dict'])
    # scheduler.load_state_dict(cp['scheduler_state_dict'])
    # counter = cp['epoch']
    model.train()

    lambda_conn_deficiency = torch.zeros(1).to(device)
    # lambda_temp = torch.ones(dataset.__len__()).to(device) * 10
    # lambda_support = torch.zeros(dataset.__len__()).to(device)
    lambda_discreteness = torch.zeros(1).to(device)  #7 这里是18.12日根据你的要求测试的几组参数PA 1e-10 PB1e-2 PC1e-3
    mu_conn_deficiency = torch.zeros(1).to(device)
    mu_discreteness = torch.zeros(1).to(device)

    discreteness_history = deque()
    conn_deficiency_history = deque()
    loss_history = deque()
    total_loss_history = deque()

    print('training started..')
    if record:
        writer = SummaryWriter(logdir=params["results_path"])

    # Maximize sum rate without any constraints
    optimizer.zero_grad()
    counter = 0
    while counter <= params["epoch"] * 8:
        for indices, channels, ue_pos in train_loader:
            #if counter == 0 or counter == params["epoch"] or counter == params["epoch"] * 2:#2 在每个阶段开始时候重制lr
            if counter % params["epoch"] == 0:
                if counter // params["epoch"] < 2:
                    lr = params["lr"]
                elif 2 <= counter // params["epoch"] < 5:
                    lr = 5 * params["lr"]
                else:
                    lr = None
                if lr is not None:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            p, deficiency = model((channels - params["mean_channel"]) / params["std_channel"])
            sr = sumrate(10 ** (channels / 10), p, params)
            loss = -sr
            # conn_penalty = None
            # discreteness_penalty = None
            if params["epoch"] * 2 >= counter > params["epoch"]:
                conn_penalty = calc_alm_penalty(deficiency, mu_conn_deficiency, lambda_conn_deficiency)
                loss += conn_penalty
            elif counter > params["epoch"] * 2:
                discreteness_penalty = calc_discreteness_penalty2(p)#0 这个函数被修改过加了1e-10
                conn_penalty = calc_alm_penalty(deficiency, mu_conn_deficiency, lambda_conn_deficiency)
                discreteness_penalty = calc_alm_penalty(discreteness_penalty, mu_discreteness, lambda_discreteness)
                loss += conn_penalty + discreteness_penalty
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)#3 这两行进行了梯度剪裁
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.9)# 同上

            optimizer.step()
            optimizer.zero_grad()
            for param in model.parameters():#4 这里加了参数剪裁
                param.data.clamp_(min=-0.5, max=0.5)
            if counter > params["epoch"]*0.5:#5 这里开了 scheduler
                scheduler.step(loss.mean().item())

            c=0#下面是history相关内容
            # for index_in_this_batch, channel_idx in enumerate(indices):
                # while len(support_history[channel_idx]) > params["patience"]:
                #     support_history[channel_idx].popleft()
                # support_history[channel_idx].append(support[index_in_this_batch].item())

            if counter > params["epoch"]:
                while len(conn_deficiency_history) > params["patience"]:
                    conn_deficiency_history.popleft()
                conn_deficiency_history.append(conn_penalty.mean().item())
                while len(loss_history) > params["patience"]:
                    loss_history.popleft()
                loss_history.append(loss.mean().item())
            if counter > params["epoch"] *2:
                while len(discreteness_history) > params["patience"]:
                    discreteness_history.popleft()
                discreteness_history.append(discreteness_penalty.mean().item())

    # 所有判断是否converged的都在这里
            converged = whether_converged(loss_history, "min", params["patience"])
            if counter > params["epoch"]:
                converged &= whether_converged(conn_deficiency_history, "min", params["patience"])
            if counter > params["epoch"] * 2:
                converged &= whether_converged(discreteness_history, "min", params["patience"])
            if converged:
                if counter > params["epoch"]:
                    loss_history.clear()
                    conn_deficiency_history.clear()
                    lambda_conn_deficiency, mu_conn_deficiency = alm_update_lambda_mu(lambda_conn_deficiency,
                                                                                      mu_conn_deficiency,
                                                                                      1,
                                                                                      conn_penalty)
                    c = 1
                if counter > params["epoch"] * 2:
                    discreteness_history.clear()
                    lambda_discreteness, mu_discreteness = alm_update_lambda_mu(lambda_discreteness,
                                                                                mu_discreteness,
                                                                                1e-1,
                                                                                conn_penalty)

            sr_test, deficiency_test, discreteness_penalty_test, compute_conn_deficiency = testing(model, dataset_testing,
                                                                                        params, counter)

            output = f"Iter={counter}, rate={sr.mean()}, compute_conn_deficiency={compute_conn_deficiency.mean()}"
            if counter > params["epoch"]:
                output += f", loss={loss.mean()}, deficiency={conn_penalty.mean()}"
                if counter > params["epoch"] * 2:
                    output += f", discreteness={discreteness_penalty_test.mean()}"
            print(output)

            if False and counter > 3 * params["epoch"] and compute_conn_deficiency.mean() == 0:#这里conn_penalty改成了compute_conn_deficiency 这个似乎没用？*3
                save_model(model, counter, params, optimizer, scheduler)
                sys.exit()
                #break
            # 所以可能需要记录的都写在这里
            data_to_record = {
                "Training/sum_rate": sr,
                "Training/loss": loss,
                "Training/lambda_conn_deficiency": lambda_conn_deficiency,
                "Training/lambda_discreteness": lambda_discreteness,
                "Training/mu_conn_deficiency": mu_conn_deficiency,
                "Training/mu_discreteness": mu_discreteness,
                "Training/conn_deficiency": deficiency,
                "Training/lr": torch.tensor([optimizer.param_groups[0]["lr"]]),
                "Testing/sum_rate": sr_test,
                "Testing/conn_deficiency": deficiency_test,
                "Testing/discreteness_penalty_test": discreteness_penalty_test,
                "Training/converged":  torch.tensor([c]),
                "compute_conn_deficiency":compute_conn_deficiency
            }
            if record and counter % 100 == 0:
                record_w(writer, counter, **data_to_record)
                save_model(model, counter, params, optimizer, scheduler)
            counter += 1
