import copy
import numpy as np
from scipy.special import logsumexp


"""
XSTAR = 0
YSTAR = 0
W = 0
SIGMA = 1.0
GAMMA = 0.9
LIM = 31
M1 = 4
M2 = 3 # previous highest
N = 100
"""
XSTAR = 1.9
YSTAR = 1.75
W = 0.1
SIGMA = 0.5287
GAMMA = 0.9
LIM = 31
M1 = 6
M2 = 4 # previous highest
N = 100


def reward(x, y, xstar=None):
    """
    immediate reward
    """
    xstar = XSTAR if xstar is None else xstar
    r = -(y - YSTAR) ** 2 - W * (x - xstar) ** 2
    return r

def myopic(canonical, precision, xstar=None):
    """
    myopic policy without change point
    """
    xstar = XSTAR if xstar is None else xstar
    covm = np.linalg.inv(precision)
    mean = covm @ canonical
    a, b = mean[0][0], mean[1][0]
    va, vb, vab = covm[0][0], covm[1][1], covm[0][1]
    x_myopic = -(vab + (a - YSTAR) * b - W * xstar) / (vb + b**2 + W)
    myopic_gain = -va - (a - YSTAR)**2 - 1 - W * xstar**2 + (vab + (a - YSTAR) * b - W * xstar)**2 / (vb + b**2 + W)
    return x_myopic, myopic_gain

def gain(canonical, precision, x, xstar=None):
    """
    expected immediate reward condition on s, x without change point
    """
    xstar = XSTAR if xstar is None else xstar
    covm = np.linalg.inv(precision)
    mean = covm @ canonical
    a, b = mean[0][0], mean[1][0]
    va, vb, vab = covm[0][0], covm[1][1], covm[0][1]
    x_gain = -va - (a - YSTAR)**2 - 1 - W * xstar**2 - (vb + b**2 + W) * x**2 - 2 * (vab + (a - YSTAR) * b - W * xstar) * x
    return x_gain

def myopic_mix(states, p, xstar=None):
    """
    myopic policy with change point
    states: posterior of previous time
    """
    xstar = XSTAR if xstar is None else xstar
    bamys, bsquare, pit = [], [], []
    for _, s in states.items():
        covm = np.linalg.inv(s["pre"])
        mean = covm @ s["can"]
        amys, b = mean[0][0] - YSTAR, mean[1][0]
        va, vb, vab = covm[0][0], covm[1][1], covm[0][1]
        bamys.append(vab + amys * b)
        bsquare.append(vb + b**2)
        pit.append(s["pit"] * (1 - p))
    pit[0] = p
    bamys = np.dot(bamys, pit)
    bsquare = np.dot(bsquare, pit)
    x_myopic = -(bamys - W * xstar) / (bsquare + W)
    myopic_gain = -va - amys**2 - 1 - W * xstar**2 + (bamys - W * xstar)**2 / (bsquare + W)
    return x_myopic, myopic_gain

def gain_mix(states, p, x, xstar=None):
    """
    expected immediate reward condition on s, x with change point
    states: posterior of previous time
    """
    xstar = XSTAR if xstar is None else xstar
    bamys, bsquare, pit = [], [], []
    for _, s in states.items():
        covm = np.linalg.inv(s["pre"])
        mean = covm @ s["can"]
        amys, b = mean[0][0] - YSTAR, mean[1][0]
        va, vb, vab = covm[0][0], covm[1][1], covm[0][1]
        bamys.append(vab + amys * b)
        bsquare.append(vb + b**2)
        pit.append(s["pit"] * (1 - p))
    pit[0] = p
    bamys = np.dot(bamys, pit)
    bsquare = np.dot(bsquare, pit)
    x_gain = -va - amys**2 - 1 - W * xstar**2 - (bsquare + W) * x**2 - 2 * (bamys - W * xstar) * x
    return x_gain

def env_response(x, alpha, beta, mean_true=None, covm_true=None, p=0, err=None):
    """
    returns:
        immediate response y and new alpha beta
    params:
        x: action
        alpha, beta: previous true values
        mean_true, covm_true: the prior of alpha beta
        p: probability of change point
        err: assign an error or use random error
    """
    ic = np.random.binomial(1, p)
    if ic:
        alpha, beta = np.random.multivariate_normal(mean_true.flatten(), covm_true)
    if err is None:
        err = np.random.normal(0.0, SIGMA, len(x)) if isinstance(x, np.ndarray) else np.random.normal(0.0, SIGMA)
    y = alpha + x * beta + err
    return y, alpha, beta

def update_without_change(canonical, precision, x, y):
    """
    update state(canonical precision) after observe x and y, no change point
    params:
        canonical, precision: state at current step
        x, y: observations
    """
    xvec = np.array([[1, x]])
    canonical_t = canonical + (xvec.T * y) / (SIGMA ** 2)
    precision_t = precision + (xvec.T @ xvec) / (SIGMA ** 2)
    return canonical_t, precision_t

def q_myopic_without_change(canonical, precision, x, xstar=None):
    """
    calculate Q function of state and action following myopic policy, no change point
    parameters:
        canonical, precision: state at current step
        x: action
    """
    totreward = np.zeros(N)
    canonical_batch = [canonical.copy() for _ in range(N)]
    precision_batch = [precision.copy() for _ in range(N)]
    xstar_batch = np.full(N, xstar)
    x_batch = np.full(N, x)
    totreward += gain(canonical, precision, x, xstar)

    covm = np.linalg.inv(precision)
    mean = covm @ canonical
    thetas = np.random.multivariate_normal(mean.flatten(), covm, size=N)
    alpha_batch = thetas[:, 0]
    beta_batch = thetas[:, 1]

    for i in range(1, LIM):
        # y_batch observation at t = i - 1
        y_batch = env_response(x_batch, alpha_batch, beta_batch)[0]
        for j in range(N):
            canonical_batch[j], precision_batch[j] = update_without_change(
                canonical_batch[j], precision_batch[j], x_batch[j], y_batch[j]
            )
            xstar_batch[j] = xstar_batch[j] if xstar is None else x_batch[j] # if xstar not None use x_{t-1}
            myp = myopic(canonical_batch[j], precision_batch[j], xstar_batch[j])
            x_batch[j] = myp[0]
            totreward[j] += (GAMMA ** i) * myp[1]
    return np.mean(totreward)

def update_with_change(states, x, y, p):
    """
    update posterior states(canonical precision logcon pit) after observe x and y, assume change points
    params:
        states: posterior of the previous time
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

def q_myopic_with_change(states, x, p, xstar=None):
    """
    calculate Q function of state and action following myopic policy, assume change point
    params:
        states: posterior of the previous time
        x: action
        p: probability of change point
    """
    estimates = np.zeros(N)
    for n in range(N):
        # initialize
        states_n = copy.deepcopy(states)
        x_n, xstar_n = x, xstar
        totgain = gain_mix(states_n, p, x_n, xstar_n)

        covm_0 = np.linalg.inv(states_n[0]["pre"])
        mean_0 = covm_0 @ states_n[0]["can"]
        if len(states_n) == 1:
            alpha_n, beta_n = np.random.multivariate_normal(mean_0.flatten(), covm_0)
        else:
            ic = np.random.choice([_ for _ in states_n.keys() if _ != 0], p=[s["pit"] for _, s in states_n.items() if _ != 0])
            covm_ic = np.linalg.inv(states_n[ic]["pre"])
            mean_ic = covm_ic @ states_n[ic]["can"]
            alpha_n, beta_n = np.random.multivariate_normal(mean_ic.flatten(), covm_ic)

        for i in range(1, LIM):
            # y_n at t = i - 1
            y_n, alpha_n, beta_n = env_response(x_n, alpha_n, beta_n, mean_0, covm_0, p)
            states_n = update_with_change(states_n, x_n, y_n, p)
            xstar_n = xstar_n if xstar is None else x_n # if xstar not None use x_{t-1}
            myp = myopic_mix(states_n, p, xstar_n)
            x_n = myp[0]
            totgain += (GAMMA ** i) * myp[1]
        estimates[n] = totgain
    return np.mean(estimates)
