import numpy as np


if __name__ == "__main__":
    f = np.load("data/15ue_20aps/data.npz")
    g = f.f.gains
    p = f.f.uepos
    g = np.transpose(g, axes=(0, 2, 1))
    np.savez("data/15ue_20aps/training_data.npz", channels=g[:1024 * 9, :, :], ue_pos=p[:1024 * 9, :, :])
    np.savez("data/15ue_20aps/testing_data.npz", channels=g[1024 * 9:, :, :], ue_pos=p[1024 * 9:, :, :])
