import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

DATA_FILE = "../../data/sp500w.csv"


def read_data():
    df = pd.read_csv(DATA_FILE)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df = df[df["Close"] > -0.2]
    return np.array(df["Close"])


class MyHMM:
    def __init__(self, num_unique_states, num_observations, seed=0):
        np.random.seed(seed)
        self.num_unique_states = num_unique_states
        self.num_observations = num_observations
        self.transition_matrix = np.ones((num_unique_states, num_unique_states))
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1)
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1).reshape(1, -1).T
        #print("A={}".format(self.transition_matrix))
        self.emission_matrix = np.zeros((num_unique_states, num_observations))
        self.initial_states_vector = np.ones((num_unique_states, 1))
        self.initial_states_vector = self.initial_states_vector / self.num_unique_states
        #print("PI={}".format(self.initial_states_vector))
        self.means = np.random.rand(self.num_unique_states)
        #print("Mean={}".format(self.means))
        #print(self.means.shape)
        self.stds = np.ones(num_unique_states)

    def create_alpha(self, data):
        alpha = np.zeros((self.num_unique_states, self.num_observations))
        for s in range(self.num_unique_states):
            alpha[s, 0] = self.initial_states_vector[s] * norm.pdf(data[0], self.means[s], self.stds[s])

        c = np.zeros(self.num_observations)
        c[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / c[0]

        for t in range(1, self.num_observations):
            for i in range(self.num_unique_states):
                for j in range(self.num_unique_states):
                    alpha[i, t] += alpha[j, t - 1] * self.transition_matrix[j, i]

                alpha[i, t] *= norm.pdf(data[t], self.means[i], self.stds[i])
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
                    beta[i, t] += beta[j, t + 1] * self.transition_matrix[i, j] * norm.pdf(data[t + 1], self.means[j],
                                                                                           self.stds[j])
            beta[:, t] = beta[:, t] / c[t + 1]

        return beta

    def e_step(self, data):
        alpha, c = self.create_alpha(data)
        #print("Alpha={}".format(alpha))
        #print("C={}".format(c))
        beta = self.create_beta(data, c)
        #print("Beta={}".format(beta))
        gamma = alpha * beta

        xi = np.zeros((self.num_unique_states, self.num_unique_states, self.num_observations))
        for t in range(1, self.num_observations):
            xi[:, :, t] = np.outer(alpha[:, t - 1], beta[:, t]) * self.transition_matrix
            xi[:, :, t] /= c[t]
            for s in range(self.num_unique_states):
                xi[:, s, t] *= norm.pdf(data[t], self.means[s], self.stds[s])
        return gamma, xi, c

    def m_step(self, data, gamma, xi):
        self.initial_states_vector = gamma[:, 0] / np.sum(gamma[:, 0])
        temp_sum = np.sum(xi[:, :, 1:], axis=2)
        self.transition_matrix = temp_sum / np.sum(temp_sum, axis=1).reshape(1, -1).T

        self.means = np.zeros(self.num_unique_states)

        gamma_sum = np.sum(gamma, axis=1)[None].T
        self.stds =  np.ones(self.num_unique_states)
        for s in range(self.num_unique_states):
            self.means[s] = np.sum(gamma[s, :] * data) / gamma_sum[s]
            xmu = data - self.means[s].reshape(1, -1).T
            xmu_xmu = np.inner(xmu, xmu)
            #xmu_xmu = xmu * xmu
            self.stds[s] = np.sum(gamma[s, :] * xmu_xmu) / gamma_sum[s] - self.means[s] * self.means[s]

    def run_main(self, data):
        print('Number of data points:', len(data))
        ll_list = []
        prev_ll = -9999999
        iter = 0
        while (True):
            iter += 1
            #print(iter)
            gamma, xi, c = hmm.e_step(data)
            #print("Gamma={}".format(gamma))
            #print("XI={}".format(xi))
            hmm.m_step(data, gamma, xi)
            #print("PI={}".format(hmm.initial_states_vector))
            #print("A={}".format(hmm.transition_matrix))
            #print("Mean={}".format(hmm.means))
            #print("STDS={}".format(hmm.stds))
            ll = np.sum(np.log(c))
            if (iter > 50 or (ll - prev_ll) < 0.05):
                print("iter={}, (ll - prev_ll)={}".format(iter, ll - prev_ll))
                break
            ll_list.append(ll)
            prev_ll = ll

        print("PI={}".format(hmm.initial_states_vector))
        print("A={}".format(hmm.transition_matrix))
        print("Mean={}".format(hmm.means))
        print("STDS={}".format(hmm.stds))
        #fig, ax = plt.subplots(figsize=(15, 4))
        #ax.plot(np.arange(len(ll_list)), ll_list, ".", color="red", label="SP500")
        #plt.show()


if __name__ == "__main__":
    data = read_data()
    #data = data[:10]
    #print(data)
    num_unique_states = 3
    num_observations = data.shape[0]
    hmm = MyHMM(num_unique_states, num_observations)
    hmm.run_main(data)
