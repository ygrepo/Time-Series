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
        alpha = np.zeros((self.num_unique_states, self.num_observations))
        for s in range(self.num_unique_states):
            alpha[s, 0] = self.initial_states_vector[s] * multivariate_normal.pdf(data[0], self.mean[:, s],
                                                                                  self.covariances[0])
        c = np.zeros(self.num_observations)
        c[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / c[0]

        for t in range(1, self.num_observations):
            for i in range(self.num_unique_states):
                for j in range(self.num_unique_states):
                    alpha[i, t] += alpha[j, t - 1] * self.transition_matrix[j, i]

                alpha[i, t] *= multivariate_normal.pdf(data[t], self.mean[:, i], self.covariances[i])
                # print("{}, {}, v1={}, alpha={}".format(t,i, v1, alpha[t, i]))
            c[t] = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / c[t]
        return alpha, c

    def create_beta(self, data, c):
        beta = np.zeros((self.num_unique_states, self.num_observations))

        for s in range(self.num_unique_states):
            beta[s, self.num_observations - 1] = 1.

        for t in range(self.num_observations - 2, -1, -1):
            for i in range(self.num_unique_states):
                for j in range(self.num_unique_states):
                    beta[i, t] += beta[j, t + 1] * \
                                  self.transition_matrix[i, j] * \
                                  multivariate_normal.pdf(data[t + 1], self.mean[:, j], self.covariances[j])
            # print(beta[t, :], c[t+1])
            beta[:, t] = beta[:, t] / c[t + 1]

        return beta

    def e_step(self, data):
        alpha, c = self.create_alpha(data)
        beta = self.create_beta(data, c)
        gamma = alpha * beta

        xi = np.zeros((self.num_unique_states, self.num_unique_states, self.num_observations))
        for t in range(1, self.num_observations):
            xi[:, :, t] = alpha[:, t - 1][None].T.dot(beta[:, t][None]) * self.transition_matrix
            xi[:, :, t] /= c[t]
            for s in range(self.num_unique_states):
                xi[:, s, t] *= multivariate_normal.pdf(data[t], self.mean[:, s], self.covariances[s])
        return gamma, xi, c

    def m_step(self, data, gamma, xi):
        self.initial_states_vector = gamma[:, 0]/ np.sum(gamma[:, 0])[None].T
        temp_sum = np.sum(xi[:, :, 1:], axis=2)
        self.transition_matrix = temp_sum / np.sum(temp_sum, axis=1)[None].T

        self.mean = np.zeros((self.num_observations, self.num_unique_states))

        gamma_sum = np.sum(gamma, axis=1)[None].T
        self.covariances = []
        for s in range(self.num_unique_states):
            p = gamma[s, :] * data
            self.mean[:,s] = np.sum(gamma[s, :] * data)/gamma_sum[s]
            X_MU = data -self.mean[:, s][None].T
            covar = X_MU.dot(((X_MU * (gamma[s, :][None])).T)) / gamma_sum[s]
            self.covariances.append(covar)
            t1 = np.outer(data, gamma[s, :] * data)
            mu_mu = np.outer(self.mean[:,s], self.mean[:,s])

    def run_main(self):
        for iter in range(10):
            gamma, xi, c = hmm.e_step(data)
            hmm.m_step(data, gamma, xi)
            ll = np.sum(np.log(c))

if __name__ == "__main__":
    data = read_data()
    data = data[:10]
    num_unique_states = 3
    num_observations = data.shape[0]
    hmm = MyHMM(num_unique_states, num_observations)
    hmm.run_main()

