# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
# 
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)
"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In mc_control, sarsa, q_learning, and double q-learning once a terminal state is reached, 
the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import numpy as np

def action_sample(state, q_s_a, eps=0.9):
    if random.random() <= eps:
        return np.argmax(q_s_a[state])
    else:
        NUM_ACTIONS = q_s_a.shape[1]
        return np.random.choice(range(NUM_ACTIONS))   

def linear_decay(start, end, iteration, max_iterations):
    step = (end - start)/max_iterations
    return start + step * iteration

def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the infinity norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 (theta parameter in the pseudocode) is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
    ### Please finish the code below ##############################################
    ###############################################################################
    for iteration in range(max_iterations):
        cur_v = v.copy()
        cur_pi = pi.copy()

        max_gap = 0

        for state in range(NUM_STATES):
            q_values = []
            for action in range(NUM_ACTIONS):
                next_state_value = 0
                for next_prob, next_state, next_reward, done in env.trans_model[state][action]:
                    next_state_value += next_prob * (next_reward + gamma * v[next_state] * (1 - done))
                q_values.append(next_state_value)
            cur_v[state] = max(q_values)
            cur_pi[state] = np.argmax(q_values)
            max_gap = max(max_gap, abs(cur_v[state] - v[state]))

        v = cur_v
        pi = cur_pi

        if max_gap < 1e-3:
            print("Early Stopping! Stop after {} iterations".format(iteration))
            break

        logger.log(iteration + 1, v, pi)
    ###############################################################################
    return pi

def policy_iteration(env, gamma, max_iterations, logger):
    """
    Optional: Implement policy iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the infinity norm 
        norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of values by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1) for _ in range(NUM_STATES)]

    # To judge if policy has converged (50 continuous same policy means convergency)
    max_policy_count = 50
    cur_policy_count = 0
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    ### Please finish the code below ##############################################
    ###############################################################################
    for iteration in range(max_iterations):
        # Policy Evaluation
        for _ in range(max_iterations):
            max_gap = 0
            cur_v = [0.0] * NUM_STATES
            for state in range(NUM_STATES):
                action = pi[state]
                for next_prob, next_state, next_reward, done in env.trans_model[state][action]:
                    cur_v[state] += next_prob * (next_reward + gamma * v[next_state] * (1 - done))
                max_gap = max(max_gap, abs(cur_v[state] - v[state]))
            v = cur_v
            if max_gap < 1e-4:
                break

        # Policy Improvement
        cur_pi = pi.copy()
        for state in range(NUM_STATES):
            q_values = []
            for action in range(NUM_ACTIONS):
                q = 0
                for next_prob, next_state, next_reward, done in env.trans_model[state][action]:
                    q += next_prob * (next_reward + gamma * v[next_state] * (1 - done))
                q_values.append(q)
            cur_pi[state] = int(np.argmax(q_values))
        logger.log(iteration + 1, v, cur_pi)
        if cur_pi == pi:
            print(f"Policy converged at iteration {iteration+1}")
            break
        pi = cur_pi
    ###############################################################################
    return pi

def on_policy_mc_control(env, gamma, max_iterations, logger):
    """
    Implement on-policy first visiti Monte Carlo control to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    cur_pi = [0] * NUM_STATES
    q_s_a = np.zeros((NUM_STATES, NUM_ACTIONS))
    n_s_a = np.zeros((NUM_STATES, NUM_ACTIONS))
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 0.3
    final_eps = 0.999 # epsilon range from 0.5 -> 0.999
    # stop exploration
    converge_estimate_parameter = 0.9
    estimated_converge_iteration = converge_estimate_parameter * max_iterations # iter > max_iter * converge_estimate_parameter, set eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.2
    final_alpha = 0.01

    ### Please finish the code below ##############################################
    ###############################################################################
    for iteration in range(max_iterations):
        # Track states, actions, and rewards for the episode
        states = []
        actions = []
        rewards = []
        state_action_pairs = []

        state = env.reset()
        done = False

        # Generate Data
        while not done:
            if iteration > estimated_converge_iteration:
                cur_eps = 1
            else:
                cur_eps = linear_decay(eps, final_eps, iteration, max_iterations)

            action = action_sample(state, q_s_a, cur_eps) # epsilon-greedy
            next_state, reward, done, info = env.step(action)

            # Store state, action, reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state_action_pairs.append((state, action))

            state = next_state

        # Calculate Return Back to Head
        G = 0

        # Iterate through the episode in reverse
        for i in reversed(range(len(states))):
            G = rewards[i] + gamma * G  # Update G with the reward and discount factor
            
            state_action_pair = (states[i], actions[i])
            # Check if it is the first visit to this state-action pair
            if state_action_pair not in state_action_pairs[:i]:
                # Update the action-value function

                # strict incremental method
                n_s_a[states[i]][actions[i]] += 1
  
                # alpah method
                cur_alpha = linear_decay(alpha, final_alpha, iteration, max_iterations)
                q_s_a[states[i]][actions[i]] += cur_alpha * (G - q_s_a[states[i]][actions[i]])

        # Update the policy
        for s in range(NUM_STATES):
            if n_s_a[s].sum() == 0:
                continue  # skip unvisited states
            best_actions = np.flatnonzero(q_s_a[s] == np.max(q_s_a[s]))
            pi[s] = np.random.choice(best_actions)
            v[s] = np.max(q_s_a[s])

        # Log current value and policy
        logger.log(iteration + 1, v, pi) 
    ###############################################################################
    return pi

def sarsa(env, gamma, max_iterations, logger):
    """
    Implement SARSA to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    cur_pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)
    q_s_a = np.zeros((NUM_STATES, NUM_ACTIONS))

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 0.3
    final_eps = 0.999 # epsilon range from 0.3 -> 0.999
    # stop exploration
    converge_estimate_parameter = 0.9
    estimated_converge_iteration = converge_estimate_parameter * max_iterations # iter > max_iter * converge_estimate_parameter, set eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.2
    final_alpha = 0.01

    ### Please finish the code below ##############################################
    ###############################################################################
    for iteration in range(max_iterations):

        # S A R S' A'
        state = env.reset()
        done = False 

        # Data Generation
        while not done:
            if iteration > estimated_converge_iteration:
                cur_eps = 1
            else:
                cur_eps = linear_decay(eps, final_eps, iteration, max_iterations)

            action = action_sample(state, q_s_a, eps = cur_eps)
            next_state, reward, done, info = env.step(action)
            next_action = action_sample(next_state, q_s_a, eps = cur_eps)
            # td error
            td_error = reward + gamma * q_s_a[next_state][next_action] - q_s_a[state][action]

            # td target
            cur_alpha = linear_decay(alpha, final_alpha, iteration, max_iterations)
            q_s_a[state][action] += cur_alpha * td_error

            state = next_state
            action = next_action

        for s in range(NUM_STATES):
            if np.sum(q_s_a[s]) == 0:
                continue  # skip unvisited states
            best_actions = np.flatnonzero(q_s_a[s] == np.max(q_s_a[s]))
            pi[s] = np.random.choice(best_actions)
            v[s] = np.max(q_s_a[s])

        logger.log(iteration + 1, v, pi) 

    ###############################################################################
    return pi

def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    q_s_a = np.zeros((NUM_STATES, NUM_ACTIONS))
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 0.3
    final_eps = 0.999 # epsilon range from 0.3 -> 0.999
    # stop exploration
    converge_estimate_parameter = 0.9
    estimated_converge_iteration = converge_estimate_parameter * max_iterations # iter > max_iter * converge_estimate_parameter, set eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.2
    final_alpha = 0.01

    ### Please finish the code below ##############################################
    ###############################################################################
    for iteration in range(max_iterations):
        state = env.reset()

        done = False
        while not done:
            if iteration > estimated_converge_iteration:
                cur_eps = 1
            else:
                cur_eps = linear_decay(eps, final_eps, iteration, max_iterations)

            action = action_sample(state, q_s_a, eps=cur_eps)
            next_state, reward, done, info = env.step(action)

            # td error
            td_error = reward + gamma * np.max(q_s_a[next_state]) - q_s_a[state][action]

            cur_alpha = linear_decay(alpha, final_alpha, iteration, max_iterations)
            q_s_a[state][action] += cur_alpha * td_error

            state = next_state

        for s in range(NUM_STATES):
            if np.sum(q_s_a[s]) == 0:
                continue  # skip unvisited states
            best_actions = np.flatnonzero(q_s_a[s] == np.max(q_s_a[s]))
            pi[s] = np.random.choice(best_actions)
            v[s] = np.max(q_s_a[s])

        logger.log(iteration + 1, v, pi) 

    ###############################################################################
    return pi

def double_q_learning(env, gamma, max_iterations, logger):
    """
    Implement double Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    q1_s_a = np.zeros((NUM_STATES, NUM_ACTIONS))
    q2_s_a = np.zeros((NUM_STATES, NUM_ACTIONS))
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 0.3
    final_eps = 0.999 # epsilon range from 0.3 -> 0.999
    # stop exploration
    converge_estimate_parameter = 0.9
    estimated_converge_iteration = converge_estimate_parameter * max_iterations # iter > max_iter * converge_estimate_parameter, set eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.2
    final_alpha = 0.01  # alpha range from 0.2 -> 0.01

    ### Please finish the code below ##############################################
    ###############################################################################
    for iteration in range(max_iterations):
        state = env.reset()

        done = False
        while not done:
            if iteration > estimated_converge_iteration:
                cur_eps = 1
            else:
                cur_eps = linear_decay(eps, final_eps, iteration, max_iterations)

            action = action_sample(state, q1_s_a + q2_s_a, eps=cur_eps)
            next_state, reward, done, info = env.step(action)

            cur_alpha = linear_decay(alpha, final_alpha, iteration, max_iterations)
            if random.random() > 0.5:
                next_action = np.argmax(q1_s_a[next_state])
                td_error = reward + gamma * (q2_s_a[next_state][next_action]) - q1_s_a[state][action]
                q1_s_a[state][action] += cur_alpha * td_error
            else:
                next_action = np.argmax(q2_s_a[next_state])
                td_error = reward + gamma * (q1_s_a[next_state][next_action]) - q2_s_a[state][action]
                q2_s_a[state][action] += cur_alpha * td_error
            state = next_state
        q_s_a = q1_s_a + q2_s_a
        
        for s in range(NUM_STATES):
            if np.sum(q_s_a[s]) == 0:
                continue  # skip unvisited states
            best_actions = np.flatnonzero(q_s_a[s] == np.max(q_s_a[s]))
            pi[s] = np.random.choice(best_actions)
            v[s] = np.max(q_s_a[s])

        logger.log(iteration + 1, v, pi) 
    ###############################################################################
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "On-policy MC Control": on_policy_mc_control,
        "SARSA": sarsa,
        "Q-Learning": q_learning,
        "Double Q-Learning": double_q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        #"world2": lambda : [
        #    [10, "s", "s", "s", 1],
        #    [-10, -10, -10, -10, -10],
        #],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()