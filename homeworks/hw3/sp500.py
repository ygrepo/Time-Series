import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

DATA_FILE = "../../data/sp500w.csv"


def read_data():
    df = pd.read_csv(DATA_FILE)
    print('Number of rows:', len(df))
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df = df[df["Close"] > -0.2]
    return np.array(df["Close"])


class MyHMM:
    def __init__(self, num_unique_states, num_observations, seed=0):
        np.random.seed(seed)
        self.num_unique_states = num_unique_states
        self.num_observations = num_observations
        self.transition_matrix = np.ones((num_unique_states, num_unique_states))
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1)[None].T
        self.emission_matrix = np.zeros((num_unique_states, num_observations))
        self.initial_states_vector = np.ones((num_unique_states, 1))
        self.initial_states_vector = self.initial_states_vector / self.num_unique_states
        self.mean = np.random.rand(self.num_observations, self.num_unique_states)
        self.covariances = [np.eye(self.num_observations) for _ in range(self.num_unique_states)]

    def create_alpha(self, data):
        alpha = np.zeros((self.num_observations, self.num_unique_states))
        for s in range(self.num_unique_states):
            alpha[0, s] = self.initial_states_vector[s] * multivariate_normal.pdf(data[0], self.mean[0],
                                                                                  self.covariances[0])
        c = np.zeros(self.num_observations)
        c[0] = np.sum(alpha[0, :])
        alpha[0, :] = alpha[0, :] / c[0]

        for t in range(1, self.num_observations):
            for i in range(self.num_unique_states):
                for j in range(self.num_unique_states):
                    alpha[t, i] = alpha[t - 1, j] * self.transition_matrix[j, i]
                alpha[t, i] *= multivariate_normal.pdf(data[t], self.mean[i, :], self.covariances[t])
                # print("{}, {}, v1={}, alpha={}".format(t,i, v1, alpha[t, i]))
            c[t] = np.sum(alpha[t, :])
            # alpha[t, :] = alpha[t, :] / c[t]
        return alpha, c

    def create_beta(self, data, c):
        beta = np.zeros((self.num_observations, self.num_unique_states))
        for s in range(self.num_unique_states):
            beta[self.num_observations - 1, s] = 1.

        for t in range(self.num_observations - 2, -1, -1):
            for i in range(self.num_unique_states):
                for j in range(self.num_unique_states):
                    beta[t, i] += beta[t + 1, j] * \
                                  self.transition_matrix[i, j] * \
                                  multivariate_normal.pdf(data[t + 1], self.mean[j, :], self.covariances[j])
            # print(beta[t, :], c[t+1])
            # beta[t, :] = beta[t, :] / c[t + 1]

        return beta

    def e_step(self, data):
        alpha, c = self.create_alpha(data)
        beta = self.create_beta(data, c)
        gamma = alpha * beta




if __name__ == "__main__":
    data = read_data()
    num_unique_states = 3
    num_observations = data.shape[0]
    hmm = MyHMM(num_unique_states, num_observations)
    alpha, c = hmm.create_alpha(data)
    hmm.create_beta(data, c)
