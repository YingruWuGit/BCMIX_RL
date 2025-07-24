import numpy as np
from scipy.special import logsumexp


XSTAR = 0
YSTAR = 0
W = 0
GAMMA = 0.9
LIM = 50
N = 200


def reward(x, y):
    """
    immediate reward from x and y
    """
    r = -(y - YSTAR) ** 2 - W * (x - XSTAR) ** 2
    return r

def myopic(canonical, precision):
    """
    myopic policy given canonical and precision
    """
    covm = np.linalg.inv(precision)
    mean = covm @ canonical
    a, b = mean[0][0], mean[1][0]
    vb, vab = covm[1][1], covm[0][1]
    myopic = -(vab + (a - YSTAR) * b - W * XSTAR) / (vb + b ** 2 + W)
    return myopic

def env_response(x, alpha, beta, mean_0=None, covm_0=None, p=0):
    """
    immediate response y and new alpha beta
    parameters:
        alpha, beta: previous values
        mean_0, covm_0: the prior of alpha beta
        p: probability of change point
    """
    if (p > 0) and (mean_0 is None) and (covm_0 is None):
        raise ValueError("If p > 0, need to provide mean_0 and covm_0")
    ic = np.random.binomial(1, p)
    if ic:
        alpha, beta = np.random.multivariate_normal(mean_0.flatten(), covm_0)
    y = alpha + x * beta + np.random.normal(0.0, 1.0)
    return y, alpha, beta

def update_without_change(canonical, precision, x, y):
    """
    update state(canonical precision) after observe x and y, assume no change point
    parameters:
        canonical, precision: state at current step
        x, y: observations
    """
    xvec = np.array([[1, x]])
    canonical_t = canonical + xvec.T * y
    precision_t = precision + xvec.T @ xvec
    return canonical_t, precision_t

def q_myopic_without_change(canonical, precision, x, alpha, beta):
    """
    given alpha beta, calculate Q function of state and action following myopic policy
    parameters:
        canonical, precision: state at current step
        x: action
    """
    totreward = np.zeros(N)
    canonical_batch = [canonical.copy() for _ in range(N)]
    precision_batch = [precision.copy() for _ in range(N)]
    x_batch = np.full(N, x)

    for i in range(LIM):
        y_batch = alpha + x_batch * beta + np.random.normal(0.0, 1.0, N)
        totreward += (GAMMA ** i) * reward(x_batch, y_batch)
        for j in range(N):
            canonical_batch[j], precision_batch[j] = update_without_change(
                canonical_batch[j], precision_batch[j], x_batch[j], y_batch[j]
            )
            x_batch[j] = myopic(canonical_batch[j], precision_batch[j])
    return np.mean(totreward)

def update_with_change(canonicals, precisions, logcons, pit, x, y, p):
    """
    update state(canonicals precisions) after observe x and y, assume there are change points
    parameters:
        canonicals, precisions: state at current step
        logcons, pit: calculated from canonicals, precisions
        x, y: observations
        p: probability of change point
    """
    # element 0 is the prior, no need to update
    canonicals_t = [canonicals[0]] + [np.nan] * len(canonicals)
    precisions_t = [precisions[0]] + [np.nan] * len(precisions)
    logcons_t = [logcons[0]] + [np.nan] * len(logcons)
    logpits_t = [float("-inf")] + [np.nan] * len(pit)
    for i in range(1, len(canonicals)):
        # update k_t = i < t
        canonicals_t[i], precisions_t[i] = update_without_change(canonicals[i], precisions[i], x, y)
        logcons_t[i] = np.linalg.slogdet(precisions_t[i])[1] / 2 - (canonicals_t[i].T @ np.linalg.inv(precisions_t[i]) @ canonicals_t[i]).item() / 2
        logpits_t[i] = np.log(1 - p) + np.log(pit[i]) + (logcons[i] - logcons_t[i])
    # calculate k_t = t
    canonicals_t[-1], precisions_t[-1] = update_without_change(canonicals[0], precisions[0], x, y)
    logcons_t[-1] = np.linalg.slogdet(precisions_t[-1])[1] / 2 - (canonicals_t[-1].T @ np.linalg.inv(precisions_t[-1]) @ canonicals_t[-1]).item() / 2
    logpits_t[-1] = np.log(p) + (logcons[0] - logcons_t[-1])
    max_logpits_t = np.max(logpits_t)
    pit_t = np.exp(np.array(logpits_t) - max_logpits_t)
    pit_t /= np.sum(pit_t)
    log_like = logsumexp(logpits_t)
    return canonicals_t, precisions_t, logcons_t, pit_t.tolist(), log_like

def q_myopic_with_change(canonicals, precisions, logcons, pit, x, alpha, beta, mean_0, covm_0, p):
    """
    given change point model and priors, calculate Q function of state and action following myopic policy
    """
    estimates = np.zeros(N)
    for _ in range(N):
        totreward = 0
        canonicals_i, precisions_i, logcons_i, pit_i, x_i = canonicals, precisions, logcons, pit, x
        alpha_i, beta_i = alpha, beta
        for i in range(LIM):
            y_i, alpha_i, beta_i = env_response(x_i, alpha_i, beta_i, mean_0, covm_0, p)
            totreward += (GAMMA ** i) * reward(x_i, y_i)
            # next state
            canonicals_i, precisions_i, logcons_i, pit_i, loglike = update_with_change(canonicals_i, precisions_i, logcons_i, pit_i, x_i, y_i, p)
            myopics = np.array([myopic(canonical, precision) for canonical, precision in zip(canonicals_i, precisions_i)])
            x_i = np.dot(myopics, pit_i)
        estimates[_] = totreward
    return np.mean(estimates)

def marginal_likelihood(p, canonicals_0, precisions_0, logcons_0, pit_0, xs, ys):
    """
    log-likelihood of the observed xs ys given canonicals_0, precisions_0, logcons_0, pit_0, as a function of p
    parameters:
        p: the variable to optimize
        canonicals_0, precisions_0, logcons_0, pit_0: priors of the change point model
        xs, ys: observed data
    """
    loglike = 0
    params_i = (canonicals_0, precisions_0, logcons_0, pit_0, 0)
    for i in range(len(xs)):
        params_i = update_with_change(*params_i[:4], xs[i], ys[i], p)
        loglike += params_i[4]
    return loglike
