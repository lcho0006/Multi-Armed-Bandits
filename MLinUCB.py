import numpy as np

from numpy import ndarray
from numpy.random import Generator
from typing import List, Deque, Optional, Tuple
from sklearn.cluster import KMeans

class MLinUCB():
    """MLinUCB

    Parameters
    ----------
    n_arms : int
        Number of arms.

    n_dims : int
        Number of features for each arm's context.

    alpha : float
        Positive real explore-exploit parameter.

    N : int
        Number of clusters

    m : int
        Number of closest clusters to use when imputing the reward.

    rng : Generator, optional
        A `Generator` used as an internal source of randomness. If None, a
        default `Generator` will be constructed using `np.random.default_rng`.
    """
    def __init__(self, n_arms: int, n_dims: int, alpha: float, N: int, m: int,
                 rng: Optional[Generator] = None) -> None:

        self.n_arms = n_arms
        self.alpha = alpha
        self.N = N
        self.m = m
        self.rng = rng
        self.A = [np.identity(n_dims) for i in range(n_arms)] #initialise A vectors for arms
        self.b = [np.zeros([n_dims,1]) for i in range(n_arms)] #initialise b vectors for arms
        self.contexts = [] #initialise list for all previous contexts
        self.rewards = [] #initialise list for all previous rewards

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

        #append current context and reward to their respective lists
        self.contexts.append(context.reshape(-1).tolist())
        self.rewards.append(reward)

        #genereate reward if nan
        if reward == None:

            #genereate clusters
            kmeans = KMeans(n_clusters=min(self.N,len(self.contexts)), random_state=0).fit(self.contexts)
            values, counts = np.unique(kmeans.labels_, return_counts=True)
            #get distances from top m clusters
            distance = kmeans.transform([context.reshape(-1).tolist()])[0]
            closest_clusters = np.argsort([distance], axis=1)[0]

            reward = 0
            m_dash = min(self.m,len(self.contexts))
            #get reward from g(x) formula
            for i in range(m_dash):
                total = 0
                for j in range(len(kmeans.labels_)):
                    if kmeans.labels_[j] == closest_clusters[i]:
                        if self.rewards[j] != None:
                            total += self.rewards[j]
                r_bar = total/counts[closest_clusters[i]]
                dist = distance[closest_clusters[i]]
                reward += (r_bar/(0.1+dist))/(1/(0.1+dist))

        #set reward at time step t to generated reward
        self.rewards[-1] = reward

        #update A and b of chosen arm with context array and reward
        x = context[arm].reshape(-1,1)
        self.A[arm] += np.dot(x, x.T)
        self.b[arm] += reward * x
