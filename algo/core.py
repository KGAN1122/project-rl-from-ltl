"""Core algorithm for Sample Efficient LTL learning
"""
import numpy as np
from itertools import product
from .mdp import MDP
import os
import importlib
import multiprocessing
import collections
import random
import torch
import numpy as np, random, torch
import torch.nn as nn, torch.optim as optim
from typing import Deque, List, Tuple, Optional
import itertools
from collections import deque
from collections import defaultdict



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T
# from torch.utils.tensorboard import SummaryWriter
# from torch.autograd import Variable

if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt

Transition = collections.namedtuple("Transition", ("state", "action", "next_state", "reward"))


class LearningAlgo:
    """
    This class is the implementation of our core algorithms.

    Attributes
    ----------
    shape : The shape of the product MDP.

    Parameters
    ----------
    mdp : The MDP model of the environment.

    auto : The automaton obtained from the LTL specification.

    discount : The discount factor.

    U : The upper bound for Max reward.

    eva_frequency: The frequency of evaluating the current policy
    """

    def __init__(self, mdp, auto, U=0.5, discount=0.99, eva_frequency=10000):
        self.mdp = mdp
        self.auto = auto
        self.discount = discount
        self.U = U  # We can also explicitly define a function of discount
        self.shape = auto.shape + mdp.shape + (len(mdp.A) + auto.shape[1],)
        

        # Create the action matrix
        self.A = np.empty(self.shape[:-1], dtype=object)
        for i, q, r, c in self.states():
            self.A[i, q, r, c] = mdp.allowed_actions((r, c)) + [len(mdp.A) + e_a for e_a in auto.eps[q]]

        self.memory = ExperienceReplay(100000)
        self.evaluation_frequency = eva_frequency

    def states(self):
        """
        Iterates through all product states
        """
        n_mdps, n_qs, n_rows, n_cols, n_actions = self.shape
        for i, q, r, c in product(range(n_mdps), range(n_qs), range(n_rows), range(n_cols)):
            yield i, q, r, c

    def random_state(self):
        """
        Generates a random product state.
        """
        n_pairs  = self.auto.shape[0] 
        n_mdps, n_qs, n_rows, n_cols, n_actions = self.shape
        mdp_state = np.random.randint(n_rows), np.random.randint(n_cols)
        return (np.random.randint(n_pairs), np.random.randint(n_qs)) + mdp_state

    # Actions = ['U', 'D', 'R', 'L']
    
    def is_accepting(self, q, label, rabin_idx=0):
        """
        General Büchi / Rabin acceptance check (supports label superset matching)
        -------------------------------------------------
        q          : Current automaton state index
        label      : Current environment label (tuple)
        rabin_idx  : Index of the Rabin acceptance set to use;
                     for single Büchi, usually 0
        -------------------------------------------------
        Returns True/False
        """
        L_cur = set(label)
        for ap_set, acc_vec in self.auto.acc[q].items():       # Iterate over all defined labels
            if set(ap_set).issubset(L_cur):                    # Superset match triggers acceptance
                if acc_vec[rabin_idx] is True:                 # This Rabin set is accepting
                    return True
        return False


    def step(self, state, action):
        """
        Performs a step in the environment from a state taking an action
        """
        i, q, r, c = state
        experiences = []
        if action < len(self.mdp.A):  # MDP actions
            mdp_states, probs = self.mdp.get_transition_prob((r, c), self.mdp.A[action])  # MDP transition
            next_state = mdp_states[np.random.choice(len(mdp_states), p=probs)]
            q_ = self.auto.delta[q][self.mdp.label[(r, c)]]  # auto transition
            # reward = self.U if self.auto.acc[q][self.mdp.label[(r, c)]][i] else 0
            label = self.mdp.label[(r, c)]
            reward = self.U if self.is_accepting(q, label, i) else 0
            experiences.append((state, action, (i, q_,) + next_state, reward))

            return (i, q_,) + next_state, experiences
        else:  # epsilon-actions
            # reward = self.U if self.auto.acc[q][self.mdp.label[r, c]][i] else 0
            label = self.mdp.label[(r, c)]
            reward = self.U if self.is_accepting(q, label, i) else 0
            experiences.append((state, action, (i, action - len(self.mdp.A), r, c), reward))
            return (i, action - len(self.mdp.A), r, c), experiences

    def counterfact_step(self, state, action, k, counterfactual):
        """
        Performs a step in the environment with counterfactual imagining
        """
        i, q, r, c = state
        experiences = []
        if action < len(self.mdp.A):  # MDP actions
            mdp_states, probs = self.mdp.get_transition_prob((r, c), self.mdp.A[action])  # MDP transition
            next_state = mdp_states[np.random.choice(len(mdp_states), p=probs)]
            q_ = self.auto.delta[q][self.mdp.label[(r, c)]]  # auto transition
            # reward = self.U if self.auto.acc[q][self.mdp.label[(r, c)]][i] else 0
            label = self.mdp.label[(r, c)]
            reward = self.U if self.is_accepting(q, label, i) else 0
            if reward:
                reward = self.U * (k + 1) / (self.K + 1)
                next_k = k + 1 if k < self.K - 1 else k
            else:
                next_k = k

            if counterfactual:
                reachable_states = set()
                for auto in range(self.shape[1]):
                    accept = self.U if self.auto.acc[auto][self.mdp.label[(r, c)]][i] else 0
                    if accept:
                        reward_ = self.U * (k + 1) / (self.K + 1)
                    else:
                        reward_ = 0
                    next_auto = self.auto.delta[auto][self.mdp.label[(r, c)]]
                    exp = (i, auto, r, c)
                    exp_ = (i, next_auto,) + next_state
                    experiences.append((exp, action, exp_, reward_))

            else:
                experiences.append((state, action, (i, q_,) + next_state, reward))
            return (i, q_,) + next_state, next_k, experiences
        else:  # epsilon-actions
            # reward = self.U * (k + 1) / self.K if self.auto.acc[q][self.mdp.label[r, c]][i] else 0
            label = self.mdp.label[(r, c)]
            reward = self.U * (k + 1) / self.K if self.is_accepting(q, label, i) else 0

            experiences.append((state, action, (i, action - len(self.mdp.A), r, c), reward))
            return (i, action - len(self.mdp.A), r, c), k, experiences


    def efficient_ltl_learning(self, start, T, trails, K, counterfactual=True):
        """
        The algorithm for sample efficient RL from LTL
        """
        T = T if T else np.prod(self.shape[:-1])
        trails = trails if trails else 100000
        self.K = K if K else 0

        print("U,K,discount", self.U, self.K, self.discount)
        Q = np.zeros(self.shape)
        for i, q, r, c in self.states():
            for a in self.A[i, q, r, c]:
                Q[i, q, r, c, a] = 2 * self.U

        epsilon = 0.1
        alpha = 0.1
        print("Q shape", Q.shape)
        probs = []
        window = 1
        for i in range(trails):
            state = (0, self.auto.q0) + (start if start else self.mdp.random_state())
            k, non_accept = 0, 0
            if (i * T) % self.evaluation_frequency == 0:
                policy = np.argmax(Q[:, :, :, :, :], axis=4)
                prob = evaluate_policy_monte_carlo(policy, self.mdp, self.auto, episodes=500, max_steps=1000)
                probs.append(prob)
                smoothed = np.convolve(probs, np.ones(window)/window, mode='valid')
                print(i, prob)

            # each episode loop
            for t in range(T):
                if np.random.rand() < epsilon or np.max(Q[state]) == 0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])
                i1, q1, r1, c1 = state
                next_state, next_k, experiences = self.counterfact_step(state, action, k, counterfactual)

                if self.auto.acc[q1][self.mdp.label[(r1, c1)]][i1]:
                    non_accept = 0
                else:
                    non_accept += 1
                if non_accept > 300:
                    break

                for exp, action, exp_, reward in experiences:
                    gamma = (1 - reward) if reward else self.discount
                    Q[exp][action] += alpha * (reward + gamma * np.max(Q[exp_]) - Q[exp][action])
                state = next_state
                k = next_k

        return Q, smoothed
    

    def efficient_ltl_learning_qmax_mb(self,
    start, T, trails, K, counterfactual=True, m=5):
        """
        Model-Based Q-max:
        - Explicitly learn the environment model (P_hat, R_hat, N)
        - Construct an optimistic MDP, assuming R_max for unknown (s, a)
        - Perform value iteration on the optimistic MDP to obtain Q_opt
        - Execute greedily, update the model, and replan at the end of each episode
        """
        # --- Parameters & Initialization ---
        T      = T      if T      else np.prod(self.shape[:-1])
        trails = trails if trails else 100000
        self.K = K      if K      else 0

        R_max = self.U  # Maximum reward bound
        γ     = self.discount

        # Environment model statistics
        N            = defaultdict(int)                    # N[(s,a)]
        counts       = defaultdict(lambda: defaultdict(int))  # counts[(s,a)][s']
        reward_sums  = defaultdict(float)                  # reward_sums[(s,a)]

        # State & action lists
        states_list  = [(i, q, r, c) for i, q, r, c in self.states()]
        action_list  = {s: self.A[s] for s in states_list}

        # --- Planning function: Perform value iteration on the optimistic MDP ---
        def plan():
            Q = np.zeros(self.shape)  # Q_opt
            tol, max_iter = 1e-4, 1000
            for _ in range(max_iter):
                Δ = 0.0
                for s in states_list:
                    for a in action_list[s]:
                        key = (s, a)
                        # If insufficiently visited, optimistically assume direct R_max → absorbing state
                        if N[key] < m:
                            q_new = R_max
                        else:
                            # Known model: compute average reward and transition probability
                            Rhat    = reward_sums[key] / N[key]
                            Pdict   = counts[key]
                            total   = sum(Pdict.values())
                            q_next  = 0.0
                            # Expected next-step value
                            for s_next, cnt in Pdict.items():
                                prob = cnt / total
                                # Value of the best action in the next state
                                best = max(Q[s_next + (a2,)] for a2 in action_list[s_next])
                                q_next += prob * best
                            q_new = Rhat + γ * q_next

                        idx = s + (a,)
                        Δ   = max(Δ, abs(q_new - Q[idx]))
                        Q[idx] = q_new
                if Δ < tol:
                    break
            return Q

        # Initial planning
        Q = plan()

        epsilon = 0.1
        probs, window = [], 1

        # --- Main loop: each episode ---
        for ep in range(trails):
            state     = (0, self.auto.q0) + (start if start else self.mdp.random_state())
            k, non_acc = 0, 0

            # Periodic evaluation
            if (ep * T) % self.evaluation_frequency == 0:
                policy = np.argmax(Q, axis=4)
                prob   = evaluate_policy_monte_carlo(
                            policy, self.mdp, self.auto,
                            episodes=500, max_steps=1000
                        )
                probs.append(prob)
                smoothed = np.convolve(probs, np.ones(window)/window, mode='valid')
                print(ep, prob)

           # Single-episode interaction
            for t in range(T):
                # Select action greedily
                action = np.argmax(Q[state])
                i1, q1, r1, c1 = state
                next_state, next_k, experiences = self.counterfact_step(
                                                    state, action, k, counterfactual
                                                )

                # Update model statistics
                for exp, a, exp_, reward in experiences:
                    key = (exp, a)
                    N[key] += 1
                    counts[key][exp_] += 1
                    reward_sums[key] += reward

                # Early termination: too many consecutive non-accepting states
                if self.auto.acc[q1][self.mdp.label[(r1, c1)]][i1]:
                    non_acc = 0
                else:
                    non_acc += 1
                if non_acc > 300:
                    break

                state, k = next_state, next_k

            # Replan on the optimistic MDP after each episode
            Q = plan()

        return Q, smoothed


    def lcrl_ql(self, start, T, trails):
        """
        The algorithm for the approach by Hasanbeig et al.
        """
        T = T if T else np.prod(self.shape[:-1])
        trails = trails if trails else 100000

        Q = np.zeros(self.shape)
        for i, q, r, c in self.states():
            for a in self.A[i, q, r, c]:
                Q[i, q, r, c, a] = 0

        epsilon = 0.1
        alpha = 0.1
        print("Q shape", Q.shape)
        probs = []
        window = 1
        for i in range(trails):
            state = (0, self.auto.q0) + (start if start else self.mdp.random_state())
            k, non_accept = 0, 0
            if (i * T) % self.evaluation_frequency == 0:
                policy = np.argmax(Q[:, :, :, :, :], axis=4)
                prob = evaluate_policy_monte_carlo(policy, self.mdp, self.auto, episodes=500, max_steps=1000)
                probs.append(prob)
                print(i, prob)
                smoothed = np.convolve(probs, np.ones(window)/window, mode='valid')

            # each episode loop
            for t in range(T):
                if np.random.rand() < epsilon or np.max(Q[state]) == 0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])
                i1, q1, r1, c1 = state
                next_state, experiences = self.step(state, action)

                if self.auto.acc[q1][self.mdp.label[(r1, c1)]][i1]:
                    non_accept = 0
                else:
                    non_accept += 1
                if non_accept > 300:
                    break

                for exp, action, exp_, reward in experiences:
                    gamma = self.discount if reward else 1
                    Q[exp][action] += alpha * (reward + gamma * np.max(Q[exp_]) - Q[exp][action])

                state = next_state

        return Q, smoothed

    # ------- LCRL-QL + STEVE -------
    def lcrl_ql_steve(self,
                    start=None,
                    T=None,
                    trails=None,
                    ensemble_size: int = 5,
                    rollout_H: int = 4,
                    batch_size: int = 128):
        """
        Based on the original lcrl_ql, augmented with the STEVE target.
        Key idea:
        1) For every newly encountered (s, a, s', rew), perform a 1-step TD update immediately;
        2) At the same time, store these transitions into replay_buf. Once the buffer
           accumulates to batch_size, train the dynamics ensemble and compute
           STEVE n-step weighted targets, then update the corresponding Q values.
        Returns Q, probs, fully compatible with lcrl_ql.
        """

        # --- STEP-0: Prepare PyTorch device / hyperparameters ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[STEVE] device = {device}")

        T      = T if T else int(np.prod(self.shape[:-1]))
        trails = trails if trails else 100000
        n_act  = len(self.mdp.A)       # Number of base MDP actions (U, D, L, R)
        obs_dim, act_dim = 2, 1
        grid_max = self.mdp.shape[0] - 1  # Max row/column index in the grid (for normalization)

        # --- STEP-1: Initialize Q and replay_buf ---
        Q = np.zeros(self.shape)
        for idx in self.states():
            for a in self.A[idx]:
                Q[idx + (a,)] = 0.0

        replay_buf = []  # Store experiences for batch updates with STEVE
        Q_target = Q.copy()
        sync_every = 200
        step_cnt   = 0
        tau = 0.05

        # --- STEP-2: Build dynamics ensemble, predicting only Δ(row, col) ---
        def make_dyn():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, obs_dim)  # Output: Δrow, Δcol
            ).to(device)

        dyn_models = [make_dyn() for _ in range(ensemble_size)]
        dyn_optims = [optim.Adam(m.parameters(), lr=1e-3) for m in dyn_models]

        # --- STEP-3: Define STEVE n-step weighted target function ---
        def steve_target(i, q, row, col, a_mdp):
            """
            Given a product-state (i, q, row, col) and an MDP action a_mdp,
            return the STEVE weighted n-step bootstrap target value
            (predicting only Δstate, not reward).
            """
            # 0-step bootstrap:
            boot0 = self.discount * np.max(Q_target[i, q, row, col, :])
            preds, precs = [boot0], [1e-6]

            # Normalize continuous coordinates to [0, 1]
            obs_h = torch.tensor([[row / grid_max, col / grid_max]], device=device)
            i_h, q_h = i, q

            for h in range(1, rollout_H + 1):
                # Normalize action to [0, 1]
                act_t = torch.tensor([[a_mdp / (n_act - 1)]], device=device)
                inp   = torch.cat([obs_h, act_t], dim=1)  # [1, 3]

                # Predict Δstate using ensemble
                deltas = torch.stack([m(inp) for m in dyn_models], dim=0)  # [N,1,2]
                mu     = deltas.mean(0)  # [1,2]
                var    = deltas.var(0).mean().item() + 1e-6  # Scalar variance

                # Sample next continuous coordinates obs_h
                eps   = torch.randn_like(mu)
                obs_h = obs_h + mu + eps * var**0.5

                # Discretize back to grid coordinates
                row_h = int(torch.clamp(obs_h[0, 0] * grid_max, 0, grid_max).item())
                col_h = int(torch.clamp(obs_h[0, 1] * grid_max, 0, grid_max).item())

                # Retrieve current AP for (row_h, col_h) and update automaton state
                props = tuple(sorted(self.mdp.label[(row_h, col_h)]))
                q_h   = self.auto.delta[q_h].get(props, self.auto.shape[1] - 1)

                # For Rabin automaton, set i_h to the index of the accepting pair
                if self.auto.shape[0] > 1:
                    for idx_acc, flag in enumerate(self.auto.acc[q_h][props]):
                        if flag is True:
                            i_h = idx_acc
                            break

                # Assign reward: if accepting state, r_hat = U; else r_hat = 0
                r_hat = self.U if any(x is True for x in self.auto.acc[q_h][props]) else 0.0
                gamma_pow = self.discount ** (h - 1)
                # n-step bootstrap: accumulate reward, then Q-bootstrap
                boot_h = gamma_pow * r_hat + (self.discount ** h) * np.max(Q_target[i_h, q_h, row_h, col_h, :])

                preds.append(boot_h)
                precs.append(1.0 / var)

                # If trapped, exit early
                if q_h == self.auto.shape[1] - 1:
                    break

            w = np.array(precs, dtype=np.float32)
            w /= w.sum()
            return float((w * np.array(preds)).sum())

        # --- STEP-4: Main training loop ---
        epsilon, alpha = 0.1, 0.3
        probs = []
        window = 20

        for ep in range(trails):
            mdp_start = start if start else self.mdp.random_state()
            state     = (0, self.auto.q0) + mdp_start  # (i, q, row, col)
            non_accept = 0

            # Evaluate current policy every evaluation_frequency steps
            if (ep * T) % self.evaluation_frequency == 0:
                policy = np.argmax(Q, axis=4)
                p = evaluate_policy_monte_carlo(
                        policy, self.mdp, self.auto,
                        episodes=500, max_steps=500)
                probs.append(p)
                smoothed = np.convolve(probs, np.ones(window)/window, mode='valid')
                print(f"[ep {ep}] MC-success = {p:.3f}")

            for _ in range(T):
                # ε-greedy action selection
                if np.random.rand() < epsilon or np.max(Q[state]) == 0:
                    a = random.choice(self.A[state])
                else:
                    a = int(np.argmax(Q[state]))

                # Execute one step in the environment
                next_state, exps = self.step(state, a)

                # — (1) Immediate 1-step TD update for each transition with reward —
                for (s_exp, a_exp, s_nxt, rew) in exps:
                    if rew:
                        gamma1 = self.discount
                    else:
                        gamma1 = 1.0
                    Q[s_exp][a_exp] += alpha * (
                        rew + gamma1 * np.max(Q_target[s_nxt]) - Q[s_exp][a_exp]
                    )

                # — (2) Store all new experiences in replay_buf for later batch update —
                replay_buf.extend(exps)

                i1, q1, r1, c1 = state
                if self.auto.acc[q1][self.mdp.label[(r1, c1)]][i1]:
                    non_accept = 0
                else:
                    non_accept += 1
                if non_accept > 500:
                    break

                # — (3) When replay_buf length >= batch_size, perform STEVE batch Q update —
                if len(replay_buf) >= batch_size:
                    batch = random.sample(replay_buf, batch_size)
                    for item in batch:
                        replay_buf.remove(item)

                    # — (3.1) Train dynamics ensemble —
                    # Only train Δ(row, col) prediction, not reward
                    idx_row = torch.tensor([s[2] for s,_,_,_ in batch],
                                            dtype=torch.float32, device=device) / grid_max
                    idx_col = torch.tensor([s[3] for s,_,_,_ in batch],
                                            dtype=torch.float32, device=device) / grid_max
                    obs     = torch.stack([idx_row, idx_col], dim=1)  # [B,2]

                    acts_mdp = torch.tensor([a_ % n_act for _,a_,_,_ in batch],
                                            dtype=torch.float32, device=device).unsqueeze(1) / (n_act - 1)

                    nxt_row = torch.tensor([ns[2] for _,_,ns,_ in batch],
                                            dtype=torch.float32, device=device) / grid_max
                    nxt_col = torch.tensor([ns[3] for _,_,ns,_ in batch],
                                            dtype=torch.float32, device=device) / grid_max
                    nxt     = torch.stack([nxt_row, nxt_col], dim=1)  # [B,2]

                    inp = torch.cat([obs, acts_mdp], dim=1)  # [B,3]
                    tgt = nxt - obs                          # [B,2]

                    for m, opt in zip(dyn_models, dyn_optims):
                        pred  = m(inp)            # [B,2]
                        loss  = ((pred - tgt) ** 2).mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    # — (3.2) Perform STEVE n-step updates for MDP actions in the batch —
                    for (s_exp, a_exp, s_nxt, rew) in batch:
                        if a_exp >= n_act:
                            continue
                        i_exp, q_exp, row_exp, col_exp = s_exp
                        # —— 1-step TD update ——
                        # Using real one-step reward and next Q
                        if rew:
                            one_step_target = rew
                        else:
                            # Standard TD target: r + γ * max_a' Q[s']
                            s1_i, s1_q, s1_r, s1_c = s_nxt
                            one_step_target = rew + self.discount * np.max(Q_target[s1_i, s1_q, s1_r, s1_c, :])
                        Q[s_exp][a_exp] += alpha * (one_step_target - Q[s_exp][a_exp])

                        # —— STEVE n-step update ——
                        if not rew:
                            a_mdp = a_exp % n_act
                            steve_t = steve_target(i_exp, q_exp, row_exp, col_exp, a_mdp)
                            diff = steve_t - Q[s_exp][a_exp]
                            if abs(diff) > 0.05:
                                Q[s_exp][a_exp] += alpha * 0.3 * diff

                state   = next_state
                epsilon = max(0.05, epsilon * 0.99)
                step_cnt += 1
                if step_cnt % sync_every == 0:
                    Q_target[:] = tau * Q + (1-tau) * Q_target

        return Q, smoothed

    def plot(self, value=None, policy=None, iq=None, **kwargs):
        """
        Plots the MDP environment with optionally the value and policy on top
        """

        if iq:
            val = value[iq] if value is not None else None
            pol = policy[iq] if policy is not None else None
            self.mdp.plot(val, pol, **kwargs)
        else:
            # A helper function for the sliders
            def plot_value(i, q):
                val = value[i, q] if value is not None else None
                pol = policy[i, q] if policy is not None else None
                self.mdp.plot(val, pol, **kwargs)

            i = IntSlider(value=0, min=0, max=self.shape[0] - 1)
            q = IntSlider(value=self.auto.q0, min=0, max=self.shape[1] - 1)
            interact(plot_value, i=i, q=q)


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # if memory isn't full, add a new experience
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
def evaluate_policy_monte_carlo(policy,
                                mdp,
                                automaton,
                                episodes: int = 200,
                                max_steps: int = 100):
    """
    Monte Carlo evaluation of policy success rate (proportion satisfying the LTL).
    ------------------------------------------------------------
    policy     ndarray of shape (1, |Q|, H, W), elements are 'action indices'
    mdp        Your MDP instance (with .transition_probs / .allowed_actions)
    automaton  OmegaAutomaton instance (with .delta / .eps / .acc)
    ------------------------------------------------------------
    Returns float ∈ [0,1]
    """
    base_actions = len(mdp.A)                 # 4 - U/D/R/L
    trap_q       = automaton.shape[1] - 1     # trap state index
    success      = 0

    for _ in range(episodes):

        s = mdp.plot_start                    # Starting point (row, col)
        q = automaton.q0                      # Initial automaton state

        for _ in range(max_steps):

            # -------- Read the AP of the current cell and convert to delta key --------
            props = tuple(sorted(mdp.label[s]))
            # ---------- Success check: current transition is accepting ----------
            acc_vec = automaton.acc[q][props]
            if any(x is True for x in acc_vec):
                success += 1
                break

            # ----------- Read action from policy -----------
            a_raw = int(policy[0, q, s[0], s[1]])

            # -------- Construct list of currently allowed actions ----------
            allowed_mdp   = mdp.allowed_actions(s)              # 0-3
            allowed_eps   = [base_actions + t for t in automaton.eps[q]]
            allowed_total = allowed_mdp + allowed_eps

            # If the policy action is illegal, fall back to the first legal action
            a = a_raw if a_raw in allowed_total else allowed_total[0]

            # ----------- Execute action -----------
            if a < base_actions:                                # Base MDP action
                next_states, probs = mdp.transition_probs[s][a]
                s = next_states[np.random.choice(len(next_states), p=probs)]

                # After an MDP action, automaton transitions once via δ using *old props*
                q = automaton.delta[q].get(props, trap_q)

            else:                                               # ε-action
                q_target = a - base_actions
                # Allowed only if in eps[q]
                if q_target in automaton.eps[q]:
                    q = q_target
                # Otherwise keep q unchanged (illegal ε-action)

            if q == trap_q:     # If trap state is reached, terminate the trajectory
                break

    return success / episodes

