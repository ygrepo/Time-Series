from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import norm

DATA_FILE = "../../data/sp500w.csv"


class ModelType(Enum):
    MYHMM = 1
    GAUSSHMM = 2
    GMMHMM = 3


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
        self.init_parameters()

    def init_parameters(self):
        self.transmat_ = np.ones((self.num_unique_states, self.num_unique_states))
        self.transmat_ = self.transmat_ / np.sum(self.transmat_, axis=1)
        self.transmat_ = self.transmat_ / np.sum(self.transmat_, axis=1).reshape(1, -1).T
        # print("A={}".format(self.transition_matrix))
        self.emission_matrix = np.zeros((self.num_unique_states, self.num_observations))
        self.startprob_ = np.ones((self.num_unique_states, 1))
        self.startprob_ = self.startprob_ / self.num_unique_states
        # print("PI={}".format(self.initial_states_vector))
        self.means_ = np.random.rand(self.num_unique_states)
        # print("Mean={}".format(self.means))
        # print(self.means.shape)
        self.covars_ = np.ones(self.num_unique_states)

    def init_parameters_tsa4(self):
        self.transmat_ = np.array([[0.945, 0.055, 0], [.739, 0, .261], [.032, .027, .942]])
        self.emission_matrix = np.zeros((self.num_unique_states, self.num_observations))
        self.startprob_ = np.ones((self.num_unique_states, 1))
        self.startprob_ = self.startprob_ / self.num_unique_states
        # print("PI={}".format(self.initial_states_vector))
        self.means_ = np.random.rand(self.num_unique_states)
        # print("Mean={}".format(self.means))
        # print(self.means.shape)
        self.covars_ = np.ones(self.num_unique_states)

    def create_alpha(self, data):
        alpha = np.zeros((self.num_unique_states, self.num_observations))
        for s in range(self.num_unique_states):
            alpha[s, 0] = self.startprob_[s] * norm.pdf(data[0], self.means_[s], self.covars_[s])

        c = np.zeros(self.num_observations)
        c[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / c[0]

        for t in range(1, self.num_observations):
            for i in range(self.num_unique_states):
                for j in range(self.num_unique_states):
                    alpha[i, t] += alpha[j, t - 1] * self.transmat_[j, i]

                alpha[i, t] *= norm.pdf(data[t], self.means_[i], self.covars_[i])
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
                    beta[i, t] += beta[j, t + 1] * self.transmat_[i, j] * norm.pdf(data[t + 1], self.means_[j],
                                                                                   self.covars_[j])
            beta[:, t] = beta[:, t] / c[t + 1]

        return beta

    def e_step(self, data):
        alpha, c = self.create_alpha(data)
        # print("Alpha={}".format(alpha))
        # print("C={}".format(c))
        beta = self.create_beta(data, c)
        # print("Beta={}".format(beta))
        gamma = alpha * beta

        xi = np.zeros((self.num_unique_states, self.num_unique_states, self.num_observations))
        for t in range(1, self.num_observations):
            xi[:, :, t] = np.outer(alpha[:, t - 1], beta[:, t]) * self.transmat_
            xi[:, :, t] /= c[t]
            for s in range(self.num_unique_states):
                xi[:, s, t] *= norm.pdf(data[t], self.means_[s], self.covars_[s])
        return gamma, xi, c

    def m_step(self, data, gamma, xi):
        self.startprob_ = gamma[:, 0] / np.sum(gamma[:, 0])
        temp_sum = np.sum(xi[:, :, 1:], axis=2)
        self.transmat_ = temp_sum / np.sum(temp_sum, axis=1).reshape(1, -1).T

        self.means_ = np.zeros(self.num_unique_states)

        gamma_sum = np.sum(gamma, axis=1)[None].T
        self.covars_ = np.ones(self.num_unique_states)
        for s in range(self.num_unique_states):
            self.means_[s] = np.sum(gamma[s, :] * data) / gamma_sum[s]
            xmu = data - self.means_[s].reshape(1, -1).T
            xmu_xmu = np.inner(xmu, xmu)
            # xmu_xmu = xmu * xmu
            self.covars_[s] = np.sum(gamma[s, :] * xmu_xmu) / gamma_sum[s] - self.means_[s] * self.means_[s]

    def fit(self, data):
        print('Number of data points:', len(data))
        ll_list = []
        prev_ll = -9999999
        iter = 0
        while (True):
            iter += 1
            # print(iter)
            gamma, xi, c = self.e_step(data)
            # print("Gamma={}".format(gamma))
            # print("XI={}".format(xi))
            self.m_step(data, gamma, xi)
            # print("PI={}".format(hmm.initial_states_vector))
            # print("A={}".format(hmm.transition_matrix))
            # print("Mean={}".format(hmm.means))
            # print("STDS={}".format(hmm.stds))
            ll = np.sum(np.log(c))
            if (iter > 50 or (ll - prev_ll) < 0.05):
                print("iter={}, (ll - prev_ll)={}".format(iter, ll - prev_ll))
                break
            ll_list.append(ll)
            prev_ll = ll


def create_model(num_states, data, model_type):
    num_observations = data.shape[0]
    transmat_prior = np.array([[0.945, 0.055, 0.00001], [.739, 0.00001, .261], [.032, .027, .942]])
    if model_type == ModelType.MYHMM:
        model = MyHMM(num_states, num_observations)
    elif model_type == ModelType.GAUSSHMM:
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
        data = data.reshape(-1, 1)
    elif model_type == ModelType.GMMHMM:
        model = hmm.GMMHMM(n_components=3, n_mix=3, covariance_type="diag")
        data = data.reshape(-1, 1)

    return model, data


def plot_gaussian(model):
    print("PI={}".format(model.startprob_))
    print("A={}".format(model.transmat_))
    print("STDS={}".format(model.covars_))

    fig, ax = plt.subplots(figsize=(15, 4))
    means = model.means_.flatten()
    print("Mean={}".format(means))
    ind = means.argsort()[-3:][::-1]
    means = means[ind]
    stds = model.covars_.flatten()
    print("Stds={}".format(stds))
    # ind = stds.argsort()[-3:][::-1]
    stds = stds[ind]
    print("Stds={}".format(stds))
    x = np.linspace(means[0] - 3 * stds[0], means[0] + 3 * stds[0], 100)
    ax.plot(x, norm.pdf(x, means[0], stds[0]), ".", color="red")
    ax.plot(x, norm.pdf(x, means[1], stds[1]), ".", color="blue")
    ax.plot(x, norm.pdf(x, means[2], stds[2]), ".", color="green")
    plt.show()


def plot_gmmhmm(model):
    fig, ax = plt.subplots(figsize=(15, 4))
    means = model.means_.flatten()
    print("means={}".format(means))
    stds = model.covars_.flatten()
    print("sigmas={}".format(stds))
    mu = means[2]
    sigma = stds[2]
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
    ax.plot(x, norm.pdf(x, mu, sigma), ".", color="red")
    ax.plot(x, 5 * norm.pdf(x, means[1], stds[1]), ".", color="blue")
    ax.plot(x, 10 * norm.pdf(x, means[2], stds[2]), ".", color="green")
    plt.show()


if __name__ == "__main__":
    data = read_data()
    model, data = create_model(3, data, ModelType.GAUSSHMM)
    model.fit(data)
    plot_gaussian(model)
    # print(model.monitor_.converged)
