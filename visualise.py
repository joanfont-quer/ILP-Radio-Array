import numpy as np
import matplotlib.pyplot as plt


def kl_divergence(p, q):
    return p * np.log(p / q)


def kl_grad(p, q):
    wrt_p = np.log(p / q) + 1
    wrt_q = -p / q
    return np.array([wrt_p, wrt_q])


def kl_approx(p, q, i, j):
    return kl_divergence(i, j) + kl_grad(i, j) @ [p - i, q - j]


print(kl_approx(0.4, 0.2, 0.4, 0.25))
print(kl_divergence(0.4, 0.2))