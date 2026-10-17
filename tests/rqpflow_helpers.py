"""Small physical system with asymmetric shunts and a phase-shifting tap."""
import numpy as np
from core.utils.pf_system import PFSystem


def rich_system():
    return PFSystem(
        np.array([1, 2, 3]),
        np.array([[.4, .1], [.3, 1.02], [0., 1.]]),
        np.array([[.01, .03], [.02, -.01], [0., .02]]),
        np.array([[0, 1], [1, 2]]),
        np.array([[.02, .20, .013, .02, .023, .04, 1.07, .13],
                  [.01, .15, 0., .01, 0., .01, 1., 0.]]))
