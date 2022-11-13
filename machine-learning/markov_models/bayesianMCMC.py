# maybe move these to a jupyter notebook
# 2022/11/13 author: Hyung Jip Lee
# just practice code, not library style

################################
# part 1: rejection sampling   #
################################

import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def rejection_sampling_example():
    # a)
    a = 2
    b = 6
    rv_p = scipy.stats.beta(a, b)
    rv_q = scipy.stats.norm(loc=0.5, scale=0.5)
    k = 5

    np.random.seed(0)
    x_q0 = rv_q.rvs(int(1e4))
    x_q = x_q0[(x_q0 >= 0) & (x_q0 <= 1)]
    crits = rv_p.pdf(x_q) / (rv_q.pdf(x_q) * k)
    coins = np.random.rand(len(x_q))
    x_p = x_q[coins < crits]

    # mean of our beta distribution is .25 since a = 2 and b = 6
    # estimation of this target mean via monte carlo integration:
    print(x_p.mean())

    plt.subplot(211)
    sns.distplot(x_q, kde=False)
    plt.title("samples from proposal distribution")
    plt.subplot(212)
    sns.distplot(x_p, kde=False)
    plt.title("samples filtered with rejection sampling")
    plt.tight_layout()
    plt.show()

    # drawback of rejection sampling:
    print((len(x_q0) - len(x_p)) / len(x_q0), " % of the samples was rejected.")

    # estimating target mean with importance sampling:
    print("from importance sampling, estimate of target mean is: ",np.mean(x_q0 * rv_p.pdf(x_q0) / rv_q.pdf(x_q0)))

    # b)
    plt.subplot(211)
    xx = np.linscipyace(0, 1, 100)
    plt.plot(xx, rv_p.pdf(xx), 'r-', label="target distribution: Beta dist.")
    plt.plot(xx, rv_q.pdf(xx) * k, 'g:', label="Proposal: Gaussian N(0.5, 0.25)")
    plt.legend()
    plt.title("Target and Proposal")
    plt.subplot(212)
    y = np.random.rand(len(x_q)) * rv_q.pdf(x_q)
    plt.plot(x_q, y, 'bs', ms=1, label="rejected samples")
    ids = coins < crits
    plt.plot(x_q[ids], y[ids], 'ro', ms=4, label="accepted samples")
    plt.legend()
    plt.title("rejected and accepted samples")
    plt.tight_layout()
    plt.show()

#########################
# part 2: markov chain  #
#########################

from dataclasses import dataclass
from numpy.linalg import matrix_power

@dataclass
class MarkovChain:
    pi: np.matrix # initial dist
    T: np.matrix # transition matrix

    def get_new_state_probas(self, steps: int):
        return self.pi * matrix_power(self.T, steps)

###############################
# part 3: metropolis-hastings #
###############################
# convert below this into jupyter notebook
import pymc3 as pm # -> package for mcmc sample creation and bayesian inference

cov = np.array([[1., 1.5], [1.5, 4]])
mu = np.array([1, -1])

with pm.Model() as model:
    x = pm.MvNormal('x', mu=mu, cov=cov, shape=(1, 2))
    step = pm.Metropolis()
    trace = pm.sample(1000, step)

import warnings
warnings.simplefilter("ignore")

pm.traceplot(trace)
plt.show()

plt.scatter(trace['x'][:, 0, 0], trace['x'][:, 0, 1])
plt.show()

# bayesian estimation w/ mcmc
theta0 = 0.7
np.random.seed(0)
x_data1 = scipy.stats.bernoulli(theta0).rvs(10)
print(x_data1)

with pm.Model() as model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    x = pm.Bernoulli('x', p=theta, observed=x_data1)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace1 = pm.sample(2000, step=step, start=start)

pm.traceplot(trace1)
plt.show()

pm.plot_posterior(trace1)
plt.xlim(0, 1)
plt.show()

pm.summary(trace1)

np.random.seed(0)
x_data2 = scipy.stats.bernoulli(theta0).rvs(500)

with pm.Model() as model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    x = pm.Bernoulli('x', p=theta, observed=x_data2)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace2 = pm.sample(1000, step=step, start=start)
pm.traceplot(trace2)
plt.show()

pm.plot_posterior(trace2)
plt.xlim(0, 1)
plt.show()
pm.summary(trace2)

# linreg with bayes?
from sklearn.datasets import make_regression

x, y_data, coef = make_regression(
    n_samples=100, n_features=1, bias=0, noise=20, coef=True, random_state=1)
x = x.flatten()

with pm.Model() as m:
    w = pm.Normal('w', mu=0, sd=50)
    b = pm.Normal('b', mu=0, sd=50)
    esd = pm.HalfCauchy('esd', 5)
    y = pm.Normal('y', mu=w * x + b, sd=esd, observed=y_data)

    start = pm.find_MAP()
    step = pm.NUTS()
    trace1 = pm.sample(10000, step=step, start=start)

pm.traceplot(trace1)
plt.show()
pm.summary(trace1)