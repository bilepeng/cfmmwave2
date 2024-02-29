import matplotlib.pyplot as plt
import tikzplotlib
import torch
from corev2 import CFData
from corev3 import GNNGumbelRecursive
from torch.utils.data import DataLoader
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from params import params


def latest_model_path(model_d): #Last model file in folder.
    model_t = [f for f in os.listdir(model_d) if f.startswith('model_')]
    latest_model = max(model_t, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(model_d, latest_model)


def load_model(m_path, device):
    if m_path.endswith('.pth') or m_path.endswith('.pt') or m_path.endswith(''):
        return torch.load(m_path, map_location=device)


def plot_ap_user(ap, ue): #Triangle "ap", gray dot for user
    plt.figure(figsize=(10, 10))
    plt.scatter(ap[:, 0], ap[:, 1], c='blue', marker='^', s=350, label='AP')
    plt.scatter(ue[0, :, 0], ue[0, :, 1], c='gray', marker='o', s=50, label='Unselected User')


def mark_user(ue, selected_users):# Red for "ap"-selected users.
    for ue_in in selected_users:
        for ue_idx in ue_in :
            if 0 <= ue_idx < ue.shape[1]:
                plt.scatter(ue[0, ue_idx, 0], ue[0, ue_idx, 1], c='red', marker='o', s=150, label= 'User')#c = [(i==j)*1 for j in range(3)]


def vis_results(ap, ue, selected_users):
    for k in range(selected_users.shape[0]):
        for ap_idx, ue_idx in enumerate(selected_users[k]):
            if ue_idx == -1:
                continue
            plt.plot([ap[ap_idx, 0], ue[0, ue_idx, 0]], [ap[ap_idx, 1], ue[0, ue_idx, 1]], c='green')

def pm(ue,dbm,params):
    f=''
    us=torch.zeros(ue.shape[1], dtype=torch.int32)
    for i in range(dbm.shape[0]):
        for ue_idx in dbm[i]:
            us[ue_idx] += 1
    for ue_idx in range(ue.shape[1]):
        if us[ue_idx] <1:
            f = '2' + f
            continue
        elif 0 < us[ue_idx] < params["min_conns_ue"]:
            f = '1' +  f
            c = 'black'
        else:
            f = '' + f
            c = 'red'
        plt.scatter(ue[0, ue_idx, 0], ue[0, ue_idx, 1], c=c, marker='o', s=50, label='User')
    return f

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--model_path", default=None)
    # parser.add_argument("--dataset_path", default='data/4ue_5aps/testing_data.npz')
    parser.add_argument("--dataset_path", default='data/15ue_20aps/testing_data.npz')
    args = parser.parse_args()

    if args.device is not None:
        device = args.device
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model_path:
        model_path = args.model_path
        model_f = os.path.basename(os.path.dirname(model_path))
    else:
        model_f = '28-02-2024_10-42-24' #Folder name of test model
        model_path = latest_model_path(os.path.join('results', model_f))

    vis_path = Path(f'results/{model_f}/vis_test')
    vis_path.mkdir(parents=True, exist_ok=True)
    print("model_path:", model_path)
    print("visualization_results:", vis_path)

    model = GNNGumbelRecursive(params, device)
    # model_w = load_model(model_path, device)
    # model.load_state_dict(model_w)
    cp = torch.load(model_path, map_location=device)
    model.load_state_dict(cp['model_state_dict'])
    #model.eval()

    data = CFData(params, path=args.dataset_path, device=device)
    test_loader = DataLoader(dataset=data, batch_size=1, shuffle=False)
    #4ue_5ap
    # ap = torch.tensor([[5., 5.], [50., 5.], [95., 5.], [5., 95.], [50., 95.]])
    #15ue_20ap
    ap = torch.tensor([[50., 50.], [275., 50.], [500., 50.], [725., 50.], [950., 50.],
                         [50., 350.], [275., 350.], [500., 350.], [725., 350.], [950., 350.],
                         [50., 650.], [275., 650.], [500., 650.], [725., 650.], [950., 650.],
                         [50., 950.], [275., 950.], [500., 950.], [725., 950.], [950., 950.]])
    # ap = ap.flip(dims=[1])

    with torch.no_grad():
        for idx, (_, channels, ue) in enumerate(test_loader):
            assignment, _ = model((channels - params["mean_channel"]) / params["std_channel"])
            ue = ue.cpu()
            plot_ap_user(ap, ue)
            for i in range(assignment.shape[1]):
                k = assignment.sum(dim=2).squeeze().round().int().max().item()
                #k = params["max_conns_ap"]
                sel_users = torch.topk(assignment[:, i, :, :], k, dim=-2).indices.squeeze()
                f = pm(ue, sel_users, params)
                #print("Selected Users:", sel_users)
                #mark_user(ue, sel_users)
                vis_results(ap, ue, sel_users)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path, f'{f}{model_f}_{idx}.png'))
            tikzplotlib.save(os.path.join(vis_path, f'{f}{model_f}_{idx}.tex'))
            #plt.savefig(os.path.join(vis_path, f'{model_f}_{idx}.png'))
            #plt.show()
            plt.close()
            #break


if __name__ == '__main__':
    main()
