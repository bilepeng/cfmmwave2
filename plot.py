import pandas as pd
from matplotlib import pyplot as plt
import tikzplotlib


if __name__ == '__main__':
    sr_training = pd.read_csv("results/28-02-2024_10-42-24/plot/run-.-tag-Training_sum_rate.csv")
    sr_testing = pd.read_csv("results/28-02-2024_10-42-24/plot/run-.-tag-Testing_sum_rate.csv")
    plt.plot(sr_training["Step"], sr_training["Value"], label="Training")
    plt.plot(sr_testing["Step"], sr_testing["Value"], label="Testing")
    plt.xlabel("Iteration")
    plt.ylabel("Sum rate (bit/Hz/s)")
    plt.legend()
    plt.grid()
    tikzplotlib.save("results/28-02-2024_10-42-24/plot/sr.tex")
    plt.close()

    sr_training = pd.read_csv("results/28-02-2024_10-42-24/plot/run-.-tag-Training_discreteness_penalty_test.csv")
    sr_testing = pd.read_csv("results/28-02-2024_10-42-24/plot/run-.-tag-Testing_discreteness_penalty_test.csv")
    plt.plot(sr_training["Step"], sr_training["Value"], label="Training")
    plt.plot(sr_testing["Step"], sr_testing["Value"], label="Testing")
    plt.xlabel("Iteration")
    plt.ylabel("Discreteness penalty")
    plt.legend()
    plt.grid()
    tikzplotlib.save("results/28-02-2024_10-42-24/plot/discreteness.tex")
    plt.close()

    sr_training = pd.read_csv("results/28-02-2024_10-42-24/plot/run-.-tag-Training_conn_deficiency.csv")
    sr_testing = pd.read_csv("results/28-02-2024_10-42-24/plot/run-.-tag-Testing_conn_deficiency.csv")
    plt.plot(sr_training["Step"], sr_training["Value"], label="Training")
    plt.plot(sr_testing["Step"], sr_testing["Value"], label="Testing")
    plt.xlabel("Iteration")
    plt.ylabel("Connection penalty")
    plt.legend()
    plt.grid()
    tikzplotlib.save("results/28-02-2024_10-42-24/plot/connection.tex")
    plt.close()

