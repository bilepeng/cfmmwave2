import numpy as np
import os


if __name__ == "__main__":
    files = os.listdir("data/chunks_20ue_20ap_5ant")
    for file in files:
        data = np.load("data/chunks_20ue_20ap_5ant/" + file)
        channels, rate_req, selection = data.f.channels, data.f.rr, data.f.sel
        channels = channels[:, :, :20, :20]
        rate_req = rate_req[:, :20]
        selection = selection[:, :, :20, :20]
        np.savez("data/chunks_20ue_20ap_5ant/" + file, channels=channels, rr=rate_req, sel=selection)

    files = os.listdir("data/chunks_20ue_20ap_5ant_testing")
    for file in files:
        data = np.load("data/chunks_20ue_20ap_5ant_testing/" + file)
        channels, rate_req, selection = data.f.channels, data.f.rr, data.f.sel
        channels = channels[:, :, :20, :20]
        rate_req = rate_req[:, :20]
        selection = selection[:, :, :20, :20]
        np.savez("data/chunks_20ue_20ap_5ant_testing/" + file, channels=channels, rr=rate_req, sel=selection)
