import copy
import numpy as np
from scipy.special import logsumexp


"""
XSTAR = 0
YSTAR = 1
W = 0
GAMMA = 0.9
LIM = 30
M1 = 4
M2 = 3
N = 100
"""
XSTAR = 1.9
YSTAR = 1.6375
W = 0.1
GAMMA = 0.9
LIM = 30
M1 = 6
M2 = 4
N = 100


def reward(x, y, xstar=None):
    """
    immediate reward from x y and xstar
    """
    xstar = XSTAR if xstar is None else xstar
    r = -(y - YSTAR) ** 2 - W * (x - xstar) ** 2
    return r

def myopic(canonical, precision, xstar=None):
    """
    myopic policy given canonical precision and xstar
    """
    xstar = XSTAR if xstar is None else xstar
    covm = np.linalg.inv(precision)
    mean = covm @ canonical
    a, b = mean[0][0], mean[1][0]
    vb, vab = covm[1][1], covm[0][1]
    x_myopic = -(vab + (a - YSTAR) * b - W * xstar) / (vb + b ** 2 + W)
    return x_myopic

def myopic_mix(states, p, xstar=None):
    """
    myopic policy for mixture model given states p(cp probability) and xstar
    """
    xstar = XSTAR if xstar is None else xstar
    myopics = np.array([myopic(s["can"], s["pre"], xstar) for _, s in states.items()])
    # note that alpha beta might change at this step
    pit_i = np.array([s["pit"] for _, s in states.items()]) * (1 - p)
    pit_i[0] = p
    x_myopic = np.dot(myopics, pit_i)
    return x_myopic

def env_response(x, alpha, beta, mean_true=None, covm_true=None, p=0, err=None):
    """
    immediate response y and new alpha beta
    parameters:
        alpha, beta: previous values
        mean_true, covm_true: the prior of alpha beta
        p: probability of change point
        err: assign an error or use random error
    """
    ic = np.random.binomial(1, p)
    if ic:
        alpha, beta = np.random.multivariate_normal(mean_true.flatten(), covm_true)
    err = np.random.normal(0.0, 1.0) if err is None else err
    y = alpha + x * beta + err
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

def q_myopic_without_change(canonical, precision, x, alpha, beta, xstar=None):
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
    xstar_batch = np.full(N, XSTAR) if xstar is None else np.full(N, xstar)

    for i in range(LIM):
        y_batch = alpha + x_batch * beta + np.random.normal(0.0, 1.0, N)
        totreward += (GAMMA ** i) * reward(x_batch, y_batch, xstar_batch)
        for j in range(N):
            canonical_batch[j], precision_batch[j] = update_without_change(
                canonical_batch[j], precision_batch[j], x_batch[j], y_batch[j]
            )
            xstar_batch[j] = XSTAR if xstar is None else x_batch[j]
            x_batch[j] = myopic(canonical_batch[j], precision_batch[j], xstar_batch[j])
    return np.mean(totreward)

def update_with_change(states, x, y, p):
    """
    update states(canonical precision logcon pit) after observe x and y, assume changes
    parameters:
        states: states at current step, a set of fixed num of mixtures, M1 latest, M2 previous highest
        x, y: observations
        p: probability of change point
    """
    states_t = copy.deepcopy(states)
    logpit_s = [float("-inf")] + [np.nan] * len(states_t)
    # kit = i < t
    for i in range(1, len(states_t)):
        states_t[i]["can"], states_t[i]["pre"] = update_without_change(states[i]["can"], states[i]["pre"], x, y)
        states_t[i]["log"] = (np.linalg.slogdet(states_t[i]["pre"])[1] - (states_t[i]["can"].T @ np.linalg.inv(states_t[i]["pre"]) @ states_t[i]["can"]).item()) / 2
        logpit_s[i] = np.log(1 - p) + np.log(states[i]["pit"]) + (states[i]["log"] - states_t[i]["log"])
    # kit = t
    states_t[-1] = {}
    states_t[-1]["can"], states_t[-1]["pre"] = update_without_change(states[0]["can"], states[0]["pre"], x, y)
    states_t[-1]["log"] = (np.linalg.slogdet(states_t[-1]["pre"])[1] - (states_t[-1]["can"].T @ np.linalg.inv(states_t[-1]["pre"]) @ states_t[-1]["can"]).item()) / 2
    logpit_s[-1] = np.log(p) + (states[0]["log"] - states_t[-1]["log"])
    max_logpit_t = np.max(logpit_s)
    pit_t = np.exp(np.array(logpit_s) - max_logpit_t)

    for i in states_t.keys():
        states_t[i]["pit"] = pit_t[i]
    if len(states_t) <= M1 + M2 + 1:
        # num mix <= bound
        states_t[len(states_t)] = states_t.pop(-1)
        ans = copy.deepcopy(states_t)
    else:
        # num mix > bound
        ans = {0: states_t[0]}
        temp = sorted([states_t[_] for _ in range(1, M2 + 2)], key=lambda s: s["pit"])
        for i in range(1, M2 + 1):
            ans[i] = temp[i]
        for i in range(M2 + 1, M2 + M1):
            ans[i] = states_t[i + 1]
        ans[M2 + M1] = states_t[-1]
    pit_sum = sum([s["pit"] for s in ans.values()])
    for i in ans.keys():
        ans[i]["pit"] /= pit_sum
    return ans

def q_myopic_with_change(states, x, alpha, beta, mean_true, covm_true, p, xstar=None):
    """
    calculate Q function of states and action following myopic policy
    parameters:
        states: states at current step
        x: action
        alpha, beta: previous env parameters, might change at this step
        mean_true, covm_true: how alpha beta generated
        p: probability of change point
    """
    estimates = np.zeros(N)
    for n in range(N):
        # initialize
        totreward = 0
        states_i, x_i, alpha_i, beta_i = copy.deepcopy(states), x, alpha, beta
        xstar_i = xstar
        for i in range(LIM):
            y_i, alpha_i, beta_i = env_response(x_i, alpha_i, beta_i, mean_true, covm_true, p)
            totreward += (GAMMA ** i) * reward(x_i, y_i, xstar_i)
            # next state
            states_i = update_with_change(states_i, x_i, y_i, p)
            xstar_i = None if xstar is None else x_i
            x_i = myopic_mix(states_i, p, xstar_i)
        estimates[n] = totreward
    return np.mean(estimates)
