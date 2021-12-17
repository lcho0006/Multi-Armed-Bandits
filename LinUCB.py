import numpy as np

from numpy import ndarray
from numpy.random import Generator
from typing import List, Deque, Optional, Tuple

class LinUCB():
    """LinUCB

    Parameters
    ----------
    n_arms : int
        Number of arms.

    n_dims : int
        Number of features for each arm's context.

    alpha : float
        Positive real explore-exploit parameter.

    rng : Generator, optional
        A `Generator` used as an internal source of randomness. If None, a
        default `Generator` will be constructed using `np.random.default_rng`.
    """
    def __init__(self, n_arms: int, n_dims: int, alpha: float,
                 rng: Optional[Generator] = None) -> None:

        self.n_arms = n_arms
        self.alpha = alpha
        self.rng = rng
        self.A = [np.identity(n_dims) for i in range(n_arms)] #initialise A vectors for arms
        self.b = [np.zeros([n_dims,1]) for i in range(n_arms)] #initialise b vectors for arms

    def play(self, context: ndarray) -> int:

        max_val = -1
        candidate_arms = []

        for arm_index in range(self.n_arms):
            A = self.A[arm_index]
            b = self.b[arm_index]
            x = context[arm_index].reshape(-1,1)

            #calculate p using formula
            A_inv = np.linalg.inv(A)
            theta = np.dot(A_inv, b)
            p = np.dot(theta.T,x) +  self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv,x)))

            #get arm with max p value
            if p > max_val:
                max_val = p
                candidate_arms = []
            #append to list for tie break
            if p == max_val:
                candidate_arms.append(arm_index)

        #pick arm uniformly from max list
        arm = np.random.choice(candidate_arms,1)[0]

        return arm

    def update(self, arm: int, context: ndarray,
               reward: Optional[float] = None) -> None:

        if reward == None:
            pass
        else:
            #update A and b of chosen arm with context array and reward
            x = context[arm].reshape(-1,1)
            self.A[arm] += np.dot(x, x.T)
            self.b[arm] += reward * x
