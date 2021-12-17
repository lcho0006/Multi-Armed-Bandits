import numpy as np

from numpy import ndarray
from numpy.random import Generator
from typing import List, Deque, Optional, Tuple
from sklearn.linear_model import LogisticRegression

class SquareCB():
    """SquareCB with a logistic regression oracle

    Parameters
    ----------
    n_arms : int
        Number of arms.

    n_dims : int
        Number of features for each arm's context.

    gamma : float
        Learning rate parameter.

    rng : Generator, optional
        A `Generator` used as an internal source of randomness. If None, a
        default `Generator` will be constructed using `np.random.default_rng`.
    """
    def __init__(self, n_arms: int, n_dims: int, gamma: float,
                 rng: Optional[Generator] = None) -> None:

        self.n_arms = n_arms
        self.gamma = gamma
        self.mu = n_arms
        self.rng = rng
        self.contexts = [[] for i in range(n_arms)] #initialise contexts for arms
        self.rewards = [[] for i in range(n_arms)] #initialise rewards for arms

    def play(self, context: ndarray) -> int:

        incomplete_arms = []
        for arm_index in range(self.n_arms):
            #add all arms without both 0 and 1 and with less than 10 entries to incomplete
            if len(set(self.rewards[arm_index]))<2 or len(self.rewards[arm_index])<10:
                incomplete_arms.append(arm_index)

        #while there are incomplete arms, uniformly choose between them
        if len(incomplete_arms) != 0:
            return np.random.choice(incomplete_arms,1)[0]

        min_val = 2
        candidate_arms = []
        yhat_list = [] #initialise yhat list for arms
        p_list = [0]*self.n_arms #initialise p value list for arms

        for arm_index in range(self.n_arms):

            #generate oracle
            clf = LogisticRegression(random_state=0).fit(self.contexts[arm_index], self.rewards[arm_index])
            #predict context to get yhat
            x = context[arm_index].reshape(1, -1)
            yhat = clf.predict(x)
            yhat = -yhat #convert reward to loss
            yhat_list.append(yhat)

            #get arm with min yhat value
            if yhat < min_val:
                min_val = yhat
                candidate_arms = []

            #append to list for tie break
            if yhat == min_val:
                candidate_arms.append(arm_index)

        #pick b uniformly from min list
        b = np.random.choice(candidate_arms,1)[0]

        for arm_index in range(self.n_arms):

            #generate p value for all arms not including b using formula
            if arm_index == b:
                pass
            else:
                p = 1/(self.mu+self.gamma*(yhat_list[arm_index]-yhat_list[b]))
                p_list[arm_index] = p.tolist()[0]

        #generate p value for b
        p_list[b] = (1 - sum(p_list))

        #pick arm with weighted p list
        arm = np.random.choice(self.n_arms,1,p=p_list)[0]

        return arm

    def update(self, arm: int, context: ndarray,
               reward: Optional[float] = None) -> None:

        #update context and reward list for arm
        self.contexts[arm].append(context[arm].reshape(-1).tolist())
        self.rewards[arm].append(reward)
