params = {
          # Scenario settings
          "num_aps": 5, # number of Access points (AP) in the network
          "num_users": 10, # number of user devices in the network
          "num_antennas": 50, # number of antennas per AP
          "preselection_ap": 10,  # number of users each AP can serve
          "square_length": 100, # the square length of the area
          "pmax": 1, # max power
          # "mean_channel": 3.4025e-11,  # 10 APs, 15 users
          # "std_channel": 8.1370e-10,  # 10 APs, 15 users
          "mean_channel": 1.5365e-9,  # 10 APs, 15 users
          "std_channel": 5.9903e-9,  # 10 APs, 15 users
          "std_rate_requirements": 0.23,
          "mean_rate_requirements": 0.4,
          "objective": "sumrate",  # power or sumrate
          "noise_power": 1.38e-23 * 300 * 500e6,  # noise power
          "preselection": "all",  # type of preselection ("all", "dom" or "gsd" are the options)
          "beamformer": "mrt", # type of beamformer ("mrt", "zf")
          "ign_th": 1.22e-14,
          "max_conns_ap": 2,
          "min_conns_ue": 2,

    # GNN settings
          "phi_local_feature_dim": 5,  # 1D for RISnet (precoding problem)
          "phi_global_feature_dim": 5,  # 1D for RISnet (precoding problem)
          "gamma_local_feature_dim": 5,  # 1D for RISnet (precoding problem)
          "gamma_global_feature_dim": 5,  # 1D for RISnet (precoding problem)
          "input_feature_dim": 3,  # 2D for InterferenceNet (power control problem)
          "phi_feature_dim": 10,  # 2D for InterferenceNet (power control problem)
          "gamma_feature_dim": 10,  # 2D for InterferenceNet (power control problem)
          "lr": 1.732e-4,  # learning rate
          "gradient_accumulation": 1,
          "min_lr": 1e-5,  # minimum learning rate in learning rate scheduler
          "epoch": 30000,  # number of interations200000
          "batch_size": 512, # batch size
          "reduced_msg_input": True, # Whether x_i is input to phi, True if no x_i
          "global_kappa": False,

          # Miscellaneous
          "data_available": True, # data set already exists on device
          "required_rates_path": "data/10_15/required_rates_training.pt", # path to access the required rates for training
          "channels_path": "data/10_15/channels_training", # path to access the channel data for training
          "positions": "data/10_15/positions", # path to access positions for visualization
          "num_data_samples": 10240, # number of drops
          "num_samples_chunks": 1024, # number of samples per chunk
          "results_path": "results/", # path to store the results
          "patience": 150,
          }

# 30 APs and 50 users. Set it to True to enable this setting
if False:
    params["num_aps"] = 30
    params["num_users"] = 50
    params["preselection_ap"] = 15
    params["mean_channel"] = 2.907e-11
    params["std_channel"] = 6.518e-10
    params["required_rates_path"] = "data/30_50/required_rates_training.pt"
    params["channels_path"] = "data/30_50/channels_training"
    params["positions"] = "data/30_50/positions"
    params["square_length"] = 1000


# 20 APs and 15 users
if True:
    params["num_aps"] = 20
    params["num_users"] = 15
    params["mean_channel"] = -128.9
    # params["std_channel"] = 10.67
    params["std_channel"] = 5

# 5 APs and 4 users
if False:
    params["num_aps"] = 5
    params["num_users"] = 4
    params["mean_channel"] = -111.6
    params["std_channel"] = 9.25*5/10.67