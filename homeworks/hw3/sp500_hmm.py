# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 13:30:28 2014
Implement EM to train an HMM for whichever dataset you used for assignment 7.
The observation probs should be as in assignment 7: either gaussian, or two 
discrete distributions conditionally independent given the hidden state.

Does the HMM model the data better than the original non-sequence model?
What is the best number of states?
@author: Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import BayesianGaussianMixture
import seaborn as sns

# Note: X and mu are assumed to be column vector
DATA_FILE = "../../data/sp500w.csv"


def read_data():
    df = pd.read_csv(DATA_FILE)
    print('Number of rows:', len(df))
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df = df[df["Close"] > -0.2]
    return df["Close"]


def normPDF(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0 / (np.math.pow((2 * np.pi), float(size) / 2) * np.math.pow(det, 1.0 / 2))
        x_mu = np.matrix(x - mu).T
        inv_ = np.linalg.inv(sigma)
        result = np.math.pow(np.math.e, -0.5 * (x_mu.T * inv_ * x_mu))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
        return -1


def initForwardBackward(X, K, d, N):
    np.random.seed(0)
    # Initialize the state transition matrix, A. A is a KxK matrix where
    # element A_{jk} = p(Z_n = k | Z_{n-1} = j)
    # Therefore, the matrix will be row-wise normalized. IOW, Sum(Row) = 1  
    # State transition probability is time independent.
    A = np.ones((K, K))
    A = A / np.sum(A, 1)[None].T
    # print("A={}".format(A))

    # Initialize the marginal probability for the first hidden variable
    # It is a Kx1 vector
    PI = np.ones((K, 1)) / K
    # print("PI={}".format(PI))

    # Initialize Emission Probability. We assume Gaussian distribution
    # for emission. So we just need to keep the mean and covariance. These 
    # parameters are different for different states.
    # Mu is dxK where kth column represents mu for kth state
    # SIGMA is a list of K matrices of shape dxd. Each element represent
    # covariance matrix for the corresponding state.
    # Given the current latent variable state, emission probability is
    # independent of time
    MU = np.random.rand(d, K)
    SIGMA = [np.eye(d) for i in range(K)]
    gmm = BayesianGaussianMixture(n_components=3, init_params="kmeans", max_iter=1500)
    gmm.fit(X.reshape(-1, 1))
    MU = gmm.means_.reshape(1, -1)
    covars = gmm.covariances_.flatten()
    for i in range(covars.size):
        SIGMA[i] = np.array([covars[i]]).reshape(1, 1)

    return A, PI, MU, SIGMA


def buildAlpha(X, PI, A, MU, SIGMA):
    # We build up Alpha here using dynamic programming. It is a KxN matrix
    # where the element ALPHA_{ij} represents the forward probability
    # for jth timestep (j = 1...N) and ith state. The columns of ALPHA are
    # normalized for preventing underflow problem as discussed in secion
    # 13.2.4 in Bishop's PRML book. So,sum(column) = 1
    # c_t is the normalizing costant
    N = np.size(X, 1)
    K = np.size(PI, 0)
    Alpha = np.zeros((K, N))
    c = np.zeros(N)

    # Base case: build the first column of ALPHA
    for i in range(K):
        Alpha[i, 0] = PI[i] * normPDF(X[:, 0], MU[:, i], SIGMA[i])

    c[0] = np.sum(Alpha[:, 0])
    Alpha[:, 0] = Alpha[:, 0] / c[0]

    # Build up the subsequent columns
    for t in range(1, N):
        for i in range(K):
            for j in range(K):
                Alpha[i, t] += Alpha[j, t - 1] * A[j, i]  # sum part of recursion
            Alpha[i, t] *= normPDF(X[:, t], MU[:, i], SIGMA[i])  # product with emission prob
        c[t] = np.sum(Alpha[:, t])
        Alpha[:, t] = Alpha[:, t] / c[t]  # for scaling factors
    return Alpha, c


def buildBeta(X, c, PI, A, MU, SIGMA):
    # Beta is KxN matrix where Beta_{ij} represents the backward probability
    # for jth timestamp and ith state. Columns of Beta are normalized using
    # the element of vector c.

    N = np.size(X, 1)
    K = np.size(PI, 0)
    Beta = np.zeros((K, N))

    # Base case: build the last column of Beta
    for i in range(K):
        Beta[i, N - 1] = 1.

    # Build up the matrix backwards
    for t in range(N - 2, -1, -1):
        for i in range(K):
            for j in range(K):
                Beta[i, t] += Beta[j, t + 1] * A[i, j] * normPDF(X[:, t + 1], MU[:, j], SIGMA[j])
        Beta[:, t] = Beta[:, t] / c[t + 1]
    return Beta


def Estep(trainSet, PI, A, MU, SIGMA):
    # The goal of E step is to evaluate Gamma(Z_{n}) and Xi(Z_{n-1},Z_{n})
    # First, create the forward and backward probability matrices
    Alpha, c = buildAlpha(trainSet, PI, A, MU, SIGMA)
    # print("Alpha={}".format(Alpha))
    # print("C={}".format(c))

    Beta = buildBeta(trainSet, c, PI, A, MU, SIGMA)
    # print("Beta={}".format(Beta))

    # Dimension of Gamma is equal to Alpha and Beta where nth column represents
    # posterior density of nth latent variable. Each row represents a state
    # value of all the latent variables. IOW, (i,j)th element represents
    # p(Z_j = i | X,MU,SIGMA) 
    Gamma = Alpha * Beta

    # Xi is a KxKx(N-1) matrix (N is the length of data seq)
    # Xi(:,:,t) = Xi(Z_{t-1},Z_{t})
    N = np.size(trainSet, 1)
    K = np.size(PI, 0)
    Xi = np.zeros((K, K, N))
    for t in range(1, N):
        Xi[:, :, t] = (1 / c[t]) * Alpha[:, t - 1][None].T.dot(Beta[:, t][None]) * A
        # Now columnwise multiply the emission prob
        for col in range(K):
            Xi[:, col, t] *= normPDF(trainSet[:, t], MU[:, col], SIGMA[col])

    return Gamma, Xi, c


def Mstep(X, Gamma, Xi):
    # Goal of M step is to calculate PI, A, MU, and SIGMA while treating
    # Gamma and Xi as constant
    K = np.size(Gamma, 0)
    d = np.size(X, 0)

    PI = (Gamma[:, 0] / np.sum(Gamma[:, 0]))[None].T
    tempSum = np.sum(Xi[:, :, 1:], axis=2)
    A = tempSum / np.sum(tempSum, axis=1)[None].T

    MU = np.zeros((d, K))
    GamSUM = np.sum(Gamma, axis=1)[None].T
    SIGMA = []
    for k in range(K):
        MU[:, k] = np.sum(Gamma[k, :] * X, axis=1) / GamSUM[k]
        X_MU = X - MU[:, k][None].T
        SIGMA.append(X_MU.dot(((X_MU * (Gamma[k, :][None])).T)) / GamSUM[k])
    return PI, A, MU, SIGMA


def plot_results(data, iter, losses, means, covars):
    fig, ax = plt.subplots(2, 1, figsize=(15, 5))
    fig.subplots_adjust(hspace=.5)

    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Log Likelihood of Data")
    ax[0].set_title("Log loss $\log{(P(X_{1:t}))}$")
    ax[0].plot(np.arange(0, iter), np.array(losses))

    # means = np.array([0.004, -0.34, -0.003])
    ind = means.argsort()[-3:][::-1]
    means = means[ind]
    print("Mean={}".format(means))
    # stds = np.array([.014, .009, .044])
    covars = covars[ind]
    print("Covars={}".format(covars))
    x1 = np.linspace(means[0] - 2 * covars[0], means[0] + 2 * covars[0], 100)
    x2 = np.linspace(means[1] - 2 * covars[1], means[1] + 2 * covars[1], 100)
    x3 = np.linspace(means[2] - 2 * covars[2], means[2] + 2 * covars[2], 100)
    sns.distplot(data, bins=50, ax=ax[1], kde=True)
    ax[1].plot(x1, norm.pdf(x1, means[0], covars[0]), ".", color="grey")
    ax[1].plot(x2, norm.pdf(x2, means[1], covars[1]), ".", color="blue")
    ax[1].plot(x3, norm.pdf(x3, means[2], covars[2]), ".", color="red")
    ax[1].set_xlabel("S&P500 Weekly Returns")
    ax[1].set_ylabel("Density")
    plt.show()


def main():
    # Reading the data file
    data = read_data()
    allData = np.array(data).reshape(-1, 1)

    allData = allData.reshape(-1, 1)
    (m, n) = np.shape(allData)

    trainSet = allData.T

    # Setting up total number of clusters which will be fixed
    K = 3

    # Initialization: Build a state transition matrix with uniform probability
    A, PI, MU, SIGMA = initForwardBackward(trainSet, K, n, m)

    iter = 0
    prev_ll = -999999
    losses = []
    while (True):
        # E-Step
        Gamma, Xi, c = Estep(trainSet, PI, A, MU, SIGMA)
        # M-Step
        PI, A, MU, SIGMA = Mstep(trainSet, Gamma, Xi)

        # Calculate log likelihood. We use the c vector for log likelihood because
        # it already gives us p(X_1^N)
        ll = np.sum(np.log(c))
        losses.append(ll)

        iter = iter + 1

        if (iter > 50 or (ll - prev_ll) < 1e-3):
            print("iter={}, (ll - prev_ll)={}".format(iter, ll - prev_ll))
            break
        print(abs(ll - prev_ll))
        prev_ll = ll

    print("PI={}".format(PI))
    print("A={}".format(A))
    print("MU={}".format(MU))
    print("SIGMA={}".format(SIGMA))
    sigmas = np.zeros(len(SIGMA))
    for i in range(len(SIGMA)):
        sigmas[i] = np.sqrt(SIGMA[i].flatten())

    plot_results(data, iter, losses, MU.flatten(), sigmas)


if __name__ == '__main__':
    main()
