from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
from scipy.stats import norm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

import homeworks.hw3.distributions as distributions
import homeworks.hw3.em as em
from homeworks.hw3 import kmeans

DATA_FILE = "../../data/sp500w.csv"


class ModelType(Enum):
    MYHMM = 1
    GAUSSHMM = 2
    GMMHMM = 3


def read_data():
    df = pd.read_csv(DATA_FILE)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df = df[df["Close"] > -0.2]
    # df = df[df["Close"] > 0]
    # df = df[df["Close"] < 25]
    return np.array(df["Close"])


class MyHMM:
    def __init__(self, num_unique_states, num_observations, data, seed=0):
        np.random.seed(seed)
        self.num_unique_states = num_unique_states
        self.num_observations = num_observations
        self.init_parameters(data)
        # self.init_parameters_with_gmm(data)

    def init_parameters(self, data):
        self.transmat_ = np.ones((self.num_unique_states, self.num_unique_states))
        self.transmat_ = self.transmat_ / np.sum(self.transmat_, axis=1)
        self.transmat_ = self.transmat_ / np.sum(self.transmat_, axis=1).reshape(1, -1).T
        self.emission_matrix = np.zeros((self.num_unique_states, self.num_observations))
        self.means_ = np.random.rand(self.num_unique_states)
        self.covars_ = np.ones(self.num_unique_states)

        # main_kmeans = cluster.KMeans(n_clusters=self.n_components,
        #                              random_state=self.random_state)
        # labels = main_kmeans.fit_predict(data)
        # kmeanses = []
        # random_state = check_random_state(None)
        # for label in range(self.n_components):
        #     kmeans = cluster.KMeans(n_clusters=self.n_mix,
        #                             random_state=self.random_state)
        #     kmeans.fit(data[np.where(labels == label)])
        #     kmeanses.append(kmeans)
        # for i, kmeans in enumerate(kmeanses):
        #     self.means_[i] = kmeans.cluster_centers_

        # Run simple EM (no HMM)
        iterations = 40
        reshaped_data = data.reshape(-1, 1)
        assignments, centers, _ = kmeans.kmeans_best_of_n(reshaped_data, self.num_unique_states, n_trials=5)
        new_centers = [distributions.Gaussian(c.mean, np.eye(1)) \
                       for c in centers]
        tau, obs_distr, pi, gmm_ll_train, gmm_ll_test = \
            em.em(reshaped_data, new_centers, assignments, n_iter=iterations)
        for i in range(len(centers)):
            self.means_[i] = centers[i].mean
        self.startprob_ = pi
        gmm = BayesianGaussianMixture(n_components=3, init_params="kmeans", max_iter=1500)
        gmm.fit(data.reshape(-1, 1))
        self.means_ = gmm.means_.flatten()
        # self.covars_ = gmm.covariances_.flatten()
        # print(self.covars_)

    def init_parameters_tsa4(self):
        # self.transmat_ = np.array([[0.945, 0.055, 0], [.739, 0, .261], [.032, .027, .942]])
        self.transmat_ = np.ones((self.num_unique_states, self.num_unique_states))
        self.transmat_ = self.transmat_ / np.sum(self.transmat_, axis=1)
        self.transmat_ = self.transmat_ / np.sum(self.transmat_, axis=1).reshape(1, -1).T
        self.emission_matrix = np.zeros((self.num_unique_states, self.num_observations))
        self.startprob_ = np.ones((self.num_unique_states, 1))
        self.startprob_ = self.startprob_ / self.num_unique_states
        # print("PI={}".format(self.initial_states_vector))
        # self.means_ = np.random.rand(self.num_unique_states)
        self.means_ = np.array([0.04, -0.34, -0.003])
        # print(self.means.shape)
        self.covars_ = np.array([.014, .009, .044])
        # self.covars_ = np.ones(self.num_unique_states)

    def init_parameters_with_gmm(self, data):
        gmm = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_process",
                                      n_components=3, init_params="random", max_iter=1500)
        gmm.fit(data.reshape(-1, 1))
        self.means_ = gmm.means_.flatten()
        # self.covars_ = gmm.covariances_.flatten()
        # print(self.covars_)
        self.covars_ = np.ones(self.num_unique_states)
        self.emission_matrix = np.zeros((self.num_unique_states, self.num_observations))
        self.startprob_ = np.ones((self.num_unique_states, 1))
        self.startprob_ = self.startprob_ / self.num_unique_states
        self.transmat_ = np.ones((self.num_unique_states, self.num_unique_states))
        self.transmat_ = self.transmat_ / np.sum(self.transmat_, axis=1)
        self.transmat_ = self.transmat_ / np.sum(self.transmat_, axis=1).reshape(1, -1).T

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
            print("Iteration={}".format(iter))
            gamma, xi, c = self.e_step(data)
            # print("Gamma={}".format(gamma))
            # print("XI={}".format(xi))
            self.m_step(data, gamma, xi)
            # print("PI={}".format(hmm.initial_states_vector))
            # print("A={}".format(hmm.transition_matrix))
            # print("Mean={}".format(hmm.means))
            # print("STDS={}".format(hmm.stds))
            ll = np.sum(np.log(c))
            if iter > 10:
                # if (iter > 40 or (ll - prev_ll) < 1e-3):
                print("iter={}, (ll - prev_ll)={}".format(iter, ll - prev_ll))
                break
            ll_list.append(ll)
            prev_ll = ll


def create_model(data, model_type, num_states=3):
    num_observations = data.shape[0]
    if model_type == ModelType.MYHMM:
        model = MyHMM(num_states, num_observations, data=data)
    elif model_type == ModelType.GAUSSHMM:
        model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=2000)
        scaler = StandardScaler()
        data = data.reshape(-1, 1)
        data = scaler.fit_transform(data.reshape(-1, 1))
    elif model_type == ModelType.GMMHMM:
        model = hmm.GMMHMM(n_components=num_states, n_mix=10, covariance_type="full")
        scaler = StandardScaler()
        data = data.reshape(-1, 1)
        data = scaler.fit_transform(data.reshape(-1, 1))

    return model, data


def plot_gaussian(data, model, num_states=3):
    print("PI={}".format(model.startprob_))
    print("A={}".format(model.transmat_))

    fig, ax = plt.subplots(figsize=(15, 4))
    means = model.means_.flatten()
    # means = np.array([0.04, -0.34, -0.003])
    # means = np.array([-0.00289348, 0.00592688, -0.00017419])
    # means = np.array([-0.00017419, 0.00592688, -0.00289348])
    ind = means.argsort()[-3:][::-1]
    means = means[ind]
    print("Mean={}".format(means))
    stds = np.sqrt(model.covars_.flatten())
    # stds = np.array([.014, .009, .044])
    # stds = np.array([0.00030478, 0.00013561, 0.00156399])
    # stds = np.array([2.25420178, 0.27484897, 1.2148339])
    # ind = stds.argsort()[-3:][::-1]
    stds = stds[ind]
    print("Stds={}".format(stds))
    # x = np.linspace(0.026, 0.07, 100)
    x = np.linspace(means[0] - 3 * stds[0], means[0] + 3 * stds[0], 100)
    sns.distplot(data, bins=25, ax=ax, kde=True)
    ax.plot(x, norm.pdf(x, means[0], stds[0]), ".", color="grey")
    ax.plot(x, norm.pdf(x, means[1], stds[1]), ".", color="blue")
    if num_states == 3:
        ax.plot(x, norm.pdf(x, means[2], stds[2]), ".", color="red")
    ax.set_xlabel("S&P500 Weekly Returns")
    ax.set_ylabel("Density")
    plt.show()


if __name__ == "__main__":
    data = read_data()

    model, data = create_model(data, ModelType.MYHMM, num_states=3)
    #model, data = create_model(data, ModelType.GAUSSHMM, num_states=3)
    model.fit(data)
    plot_gaussian(data, model, num_states=3)
    # print(model.monitor_.converged)
