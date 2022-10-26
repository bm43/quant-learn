from dataclasses import dataclass, field
from typing import List
import numpy as np
# implemented using:
# https://web.stanford.edu/~jurafsky/slp3/A.pdf

# 1.
# 2.
# 3. forward-backward algo

@dataclass
class HiddenMarkovModel():
    # n = no. of hidden states
    # m = number of possible observation outcomes

    # declare inputs to class
    transition_prob: np.ndarray # (n+2) x (n+2) input
    emission_prob: np.ndarray # m x n

    def __post_init__(self) -> None:
        self.n = self.emission_prob.shape[1]
        self.m = self.emission_prob.shape[0]

        # observation labels:
        self.obs = None

        self.forward_last = np.array([0,0])
        self.backward_last = np.array([0,0])

        self.state_proba_dict = {}

        return
    
    def _set_emission_ref(self):
        T = len(self.obs)
        keys = np.unique(self.obs)
        vals = [i for i in range(T)]
        self.emission_ref = dict(zip(keys, vals))
        return

    def fit(self, obs: List[str], iter: int = 10):
        self.obs = obs
        self._set_emission_ref()
        self.psi = np.zeros((self.n, self.n, len(self.obs)-1))
        self.gamma = np.zeros((len(self.obs), self.n, 1))
        for i in range(iter):
            old_transition = self.transition_prob.copy()
            old_emission = self.emission_prob.copy()
            print("Iteration: {}".format(i + 1))
            # perform em
            self._e_step()
            self._m_step()
        return
    
    def _e_step(self) -> None:
        self._forward = self._forward_recursion(len(self.obs))
        self._backward = self._backward_recursion(0)
        self._get_gamma() # gamma is no. of states -1 x no. of observations
        self._get_psi()
        return

    def _forward_recursion(self, idx):
        # init at idx 0
        if idx == 0:
            #fwd = [[0.0] * (len(self.obs)) for i in range(self.n)]
            fwd = np.zeros((self.n, len(self.obs), 1))
            for state in range(self.n):
                fwd[state][idx] = self._forward_init(self.obs[idx], state)
            return fwd
        # repeat
        else:
            fwd = self._forward_recursion(idx-1)
            for state in range(self.n):
                if idx != len(self.obs):
                    fwd[state][idx] = self._forward_proba(idx, fwd, state)
                else:
                    # end
                    self.forward_last[state] = self._forward_proba(idx, fwd, state, last=True)
            return fwd

    def _forward_init(self, obs, state):
        # function that gets initial forward proba
        # state obs likelihood:
        bj_ot = self.emission_prob[self.emission_ref[obs]][state] # b_j(o_t)

        return self.transition_prob[state+1][0] * bj_ot

    def _forward_proba(self, idx, forward, state, last=False):
        # = a_t
        # cf. p.6 A.12
        p = np.zeros((self.n, 1))
        for prevstate in range(self.n):
            if not last:
                # recurse
                obs_idx = self.emission_ref[self.obs[idx]]
                p[prevstate] = forward[prevstate][idx-1] * self.transition_prob[state+1][prevstate+1] * self.emission_prob[obs_idx][state]
        return sum(p)

    def _get_gamma(self) -> None:
        #print(self._forward[0,:].shape)
        #print(self._backward[0,:].shape)
        #print(self.gamma[:,0].shape)
        #print(self.gamma.shape)

        # first column:
        self.gamma[:, 0] = self._forward[0, :] * self._backward[0, :] \
            / self._forward[0, :] * self._backward[0, :] \
                + self._forward[1, :] * self._backward[1, :]
        
        # second column:
        self.gamma[:, 1] = self._forward[1, :] * self._backward[1, :] \
            / self._forward[0, :] * self._backward[0, :] \
                + self._forward[1, :] * self._backward[1, :]

        #print(self.gamma)
        return


    # could make better with matmul
    def _get_psi(self):
        # psi is N N x T matrices.
        for i in range(self.n):
            for j in range(self.n):
                for t in range(1, len(self.obs)):
                    self.psi[i][j][t-1] = self._compute_psi(t, i, j)
        
    def _compute_psi(self, t, i, j):
        alpha_i_tminus1 = self._forward[i][t-1]
        a_i_j = self.transition_prob[j+1][i+1]
        beta_t_j = self._backward[j][t]
        o_t = self.obs[t]
        b_j = self.emission_prob[self.emission_ref[o_t]][j]
        d = float(self._forward[0][i] * self._backward[0][i] + \
            self._forward[1][i] * self._backward[1][i])
        return (alpha_i_tminus1 * a_i_j * beta_t_j * b_j) / d

    def _m_step(self) -> None:
        self._get_state_probas()
        for s in range(self.n):
            self.transition_prob[s+1][0] = self.gamma[0][s]
            self.transition_prob[-1][s+1] = self.gamme[-1][s] / self.state_probas[s]
            for s2 in range(self.n):
                self.transition_prob[s2 + 1][s+1] = self._estimate_transition(s, s2)
            for o in range(self.m):
                self.emission_prob[o][s] = self._estimate_emission(s, o)
        return

    def _get_state_probas(self) -> dict:
        # proba of a state to occur
        self.state_probas = np.zeros((1, self.n))
        total_probas = list(np.sum(self.gamma, axis=0)) # column wise sum
        self.state_proba_dict = dict(zip(self.emission_ref.keys(), total_probas))
        return self.state_proba_dict

    def _estimate_transition(self, i, j) -> float:
        return sum(self.psi[i][j])
    
    def _estimate_emission(self, j, obs):
        obs = self.obs[obs]
        ts = [i for i in range(len(self.obs)) if self.obs[i] == obs]
        for i in range(len(ts)):
            ts[i] = self.gamma[ts[i]][j]
        return sum(ts) / self.state_probas[j]

    def _backward_recursion(self, idx) -> np.ndarray:
        # init at T
        if idx == (len(self.obs) - 1):
            bwd = np.zeros((self.n, len(self.obs), 1))
            for state in range(self.n):
                bwd[state][idx] = self._backward_init(state)
            return bwd
        else:
            bwd = self._backward_recursion(idx+1)
            for state in range(self.n):
                if idx >= 0:
                    bwd[state][idx] = self._backward_proba(idx, bwd, state, False)
                if idx == 0:
                    self.backward_last[state] = self._backward_proba(idx, bwd, 0, last=True)
            return bwd

    def _backward_init(self, state):
        return self.transition_prob[self.n + 1][state + 1]

    def _backward_proba(self, idx, backward, state, last) -> float:
        # backward proba at t
        p = np.zeros((self.n, 1))
        for s in range(self.n):
            observation = self.obs[idx + 1]
            if not last:
                a = self.transition_prob[s+1][state+1]
            else:
                a = self.transition_prob[s+1][0]
            b = self.emission_prob[self.emission_ref[observation]][s]
            Beta = backward[s][idx+1]
            p[s] = a * b * Beta
        return sum(p)

    def likelihood(self, current_obs) -> float:
        # proba of obseravtion seq given the current model params
        current_model = HiddenMarkovModel(self.transition_prob, self.emission_prob)
        current_model.obs = current_obs
        current_model._set_emission_ref()
        current_model._forward_recursion(len(current_obs))
        return sum(current_model.forward_last)

"""
if __name__ == "__main__":
    emission = np.array([[0.7, 0], [0.2, 0.3], [0.1, 0.7]])
    transition = np.array([ [0, 0, 0, 0], [0.5, 0.8, 0.2, 0], [0.5, 0.1, 0.7, 0], [0, 0.1, 0.1, 0]])
    obs = ['2','3','3','2','3','2','3','2','2','3','1','3','3','1','1',
                    '1','2','1','1','1','3','1','2','1','1','1','2','3','3','2',
                    '3','2','2']
    model = HiddenMarkovModel(transition, emission)
    model.fit(obs)
    #model._get_state_probas()
"""