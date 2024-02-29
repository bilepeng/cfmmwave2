import pandas as pd
from matplotlib import pyplot as plt
import tikzplotlib


if __name__ == '__main__':
    sr_training = pd.read_csv("results/big_scenario/plot/run-.-tag-Training_sum_rate.csv")
    sr_testing = pd.read_csv("results/big_scenario/plot/run-.-tag-Testing_sum_rate.csv")
    plt.plot(sr_training["Step"], sr_training["Value"], label="Training")
    plt.plot(sr_testing["Step"], sr_testing["Value"], label="Testing")
    plt.xlabel("Iteration")
    plt.ylabel("Sum rate (bit/Hz/s)")
    plt.legend()
    plt.grid()
    tikzplotlib.save("results/big_scenario/plot/sr.tex")
    plt.close()

    sr_training = pd.read_csv("results/big_scenario/plot/run-.-tag-Training_discreteness_penalty.csv")
    sr_testing = pd.read_csv("results/big_scenario/plot/run-.-tag-Testing_discreteness_penalty.csv")
    plt.plot(sr_training["Step"], sr_training["Value"], label="Training")
    plt.plot(sr_testing["Step"], sr_testing["Value"], label="Testing")
    plt.xlabel("Iteration")
    plt.ylabel("Discreteness penalty")
    plt.legend()
    plt.grid()
    tikzplotlib.save("results/big_scenario/plot/discreteness.tex")
    plt.close()

    sr_training = pd.read_csv("results/big_scenario/plot/run-.-tag-Training_conn_deficiency.csv")
    sr_testing = pd.read_csv("results/big_scenario/plot/run-.-tag-Testing_conn_deficiency.csv")
    plt.plot(sr_training["Step"], sr_training["Value"], label="Training")
    plt.plot(sr_testing["Step"], sr_testing["Value"], label="Testing")
    plt.xlabel("Iteration")
    plt.ylabel("Connection penalty")
    plt.legend()
    plt.grid()
    tikzplotlib.save("results/big_scenario/plot/connection.tex")
    plt.close()

