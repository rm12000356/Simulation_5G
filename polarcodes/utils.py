#!/usr/bin/env python



import numpy as np

def bit_reversed(x, n):

    result = 0
    for i in range(n):  # for each bit number
        if (x & (1 << i)):  # if it matches that bit
            result |= (1 << (n - 1 - i))  # set the "opposite" bit in result
    return result

def logdomain_sum(a, b):
    a = np.clip(a, -700, 700)  # FIXED: Clip for exp stability
    b = np.clip(b, -700, 700)
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))

def logdomain_diff(a, b):
    a = np.clip(a, -700, 700)
    b = np.clip(b, -700, 700)
    if a > b:
        return a + np.log1p(-np.exp(b - a))
    else:
        return b + np.log1p(-np.exp(a - b))

def bit_perm(x, p, n):

    result = 0
    for i in range(n):
        b = (x >> p[i]) & 1
        result ^= (-b ^ result) & (1 << (n - i - 1))
    return result

# find hamming weight of an index x
def hamming_wt(x, n):

    m = 1
    wt = 0
    for i in range(n):
        b = (x >> i) & m
        if b:
            wt = wt + 1
    return wt

# sort by hamming_wt()
def sort_by_wt(x, n):
    

    wts = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        wts[i] = hamming_wt(x[i], n)
    mask = np.argsort(wts)
    return x[mask]

def inverse_set(F, N):

    n = int(np.log2(N))
    not_F = []
    for i in range(N):
        if i not in F:
            not_F.append(i)
    return np.array(not_F)

def subtract_set(X, Y):

    X_new = []
    for x in X:
        if x not in Y:
            X_new.append(x)
    return np.array(X_new)

def arikan_gen(n):

    F = np.array([[1, 1], [0, 1]])
    F_n = F
    for i in range(n - 1):
        F_n = np.kron(F, F_n)
    return F_n

# Gaussian Approximation helper functions:

def phi_residual(x, val):
    return phi(x) - val

def phi(x):
    if x < 10:
        y = -0.4527 * (x ** 0.86) + 0.0218
        y = np.exp(y)
    else:
        y = np.sqrt(3.14159 / x) * (1 - 10 / (7 * x)) * np.exp(-x / 4)
    return y

def phi_inv(y):
    return bisection(y, 0, 10000)

def bisection(val, a, b):
    c = a
    while (b - a) >= 0.01:
        # check if middle point is root
        c = (a + b) / 2
        if (phi_residual(c, val) == 0.0):
            break

        # choose which side to repeat the steps
        if (phi_residual(c, val) * phi_residual(a, val) < 0):
            b = c
        else:
            a = c
    return c

def logQ_Borjesson(x):
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    if x < 0:
        x = -x
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
        y = np.log(1 - np.exp(y))
    else:
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return y
