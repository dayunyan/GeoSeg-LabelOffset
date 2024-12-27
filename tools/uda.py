import numpy as np
import torch
import torch.nn as nn


def prob_2_entropy(prob):
    """convert probabilistic prediction maps to weighted self-information maps"""
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
