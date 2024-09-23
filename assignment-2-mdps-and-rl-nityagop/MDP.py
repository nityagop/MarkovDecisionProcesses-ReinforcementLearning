import json
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matrix_mdp
import sys
import matplotlib.pyplot as plt

#######################################
# 1. Initialize transition and reward matrices
# 2. Fill transition and reward matrices with correct values
#######################################

num_states = 5
num_actions = 4

# Create transition and reward matrices below:

T = np.zeros((num_states, num_states, num_actions))
R = np.zeros((num_states, num_states, num_actions))
# Set the entries in T and R below as per the MDP used in the assignment diagram:
p = q = r = 0.5

# I used ChatGPT to help further understand how to find values for 
# each matrix and the purpose of those values 

# T 
T[0, 0, 0] = p
T[1, 0, 0] = 1 - p

T = [
    [[0.5, 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.5],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.]],

    [[0.5, 0.5, 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.]],

    [[0., 0.5, 0., 0.],
     [0., 0., 1/3, 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.]],

    [[0., 0., 0., 0.],
     [0., 0., 1/3, 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.]],

    [[0., 0., 0., 0.],
     [0., 0., 1/3, 0.],
     [0., 0., 0., 0.5],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.]]
]

T = np.array(T)

 

# R
R[4,1,2] = 10
R[4,2,3] = 10


#######################################
# 3. Map indices to action-labels 
#######################################

A = {0: "AI", 1: "A2", 2: "A3", 3: "A4", 4: "A5"}  # map each index to a human-readable action-label, such as "A1", "A2", etc. 


#######################################
# Initialize the gymnasium environment
#######################################

# No change required in this section

P_0 = np.array([1, 0, 0, 0, 0])    # This is simply the initial probability distribution which denotes where your agent is, i.e. the start state.

env=gym.make('matrix_mdp/MatrixMDP-v0', p_0=P_0, p=T, r=R)


#######################################
# 4. Random exploration
#######################################

'''
First, we reset the environment and get the initial observation.

The observation tells you which state you are in - in this case, indices 0-4 map to states S1 - S5.

Since we set P_0 = [1, 0, 0, 0, 0], the initial state is always S1 after env.reset() is called.
'''

observation, info = env.reset()

'''
Below, complete the function for random exploration, i.e. randomly choosing an action at each time-step and executing it.

A random action is simply a random integer between 0 and the number of actions (num_actions not inclusive).
However, you should make sure that the chosen action can actually be taken from the current state.
If it is not a legal move, generate a new random move. You can use the transition probabilities to figure out
whether a given action is valid from the current state.

Avoid hardcoding actions even for states where there is only one action available. That way, your
code is more general, and may be easily adapted to a different environment later.

You will use the following line of code to explore at each time step:

observation, reward, terminated, truncated, info = env.step(action)

The above line of code is used to take one step in the environment using the chosen action.
It takes as input the action chosen by the agent, and returns the next observation (i.e., state),
reward, whether the episode terminated (terminal states), whether the episode was 
truncated (max iterations reached), and additional information.

If at any point the episode is terminated (this happens when we reach a terminal state, 
and the env.step() function returns True for terminated), you should
end the episode in order to reset the environment, and start a new one.

Keep track of the total reward in each episode, and reset the environment when the episode terminates.

Print the average reward obtained over 10000 episodes. 

'''

def random_exploration():
    temp_reward = 0
    for i in range(10000):
        observation, info = env.reset()
        while True:
            if num_actions == 0:
                break
            else:
                action = np.random.choice(num_actions)
                # Used ChatGPT to find how to check for valid actions 
                while (np.sum(T[:, observation, action]) == 0):
                    action = np.random.randint(num_actions)
                observation, reward, terminated, truncated, info = env.step(action)
                temp_reward += reward
                if terminated:
                    break
                if truncated:
                    observation, info = env.reset()
            
    avg_reward = temp_reward/10000
    return avg_reward


#######################################
# 5 - 7. Policy evaluation 
#        & Plotting V_pi(s)
#######################################

gamma = 0.9

'''
Fill in the following function to evaluate a given policy.

The policy is a dictionary that maps each state to an action.
The key is an integer representing the state, and the value is an integer representing the action.

Initialize the value function V(s) = 0 for all states s.
Perform the Bellman update iteratively to update the value function.
Plot the value of S1 over time (i.e., at each iteration of Bellman update).
Save and insert this plot into the written submission.

Function returns the expected values for the first two states after all iterations.

Functions are called within main function at the end of the file.

'''

# used ChatGPT to debug some errors occuring from autograder
def evaluate_policy(policy, num_iterations, plot_filename):
    #policy = {0: 1, 1:2, 2:3}
    s1_values = [] 

    V = np.zeros(num_states)
    V_prime = np.copy(V)   

    for i in range(num_iterations): 
        s1_values.append(V_prime[0])
        for state in [0, 1, 2]:
            current_policy = policy[state]
            # used ChatGPT to find the correct syntax for accessing the values in 
            # the transition matrix 
            V_prime[state] = np.sum(T[:, state, current_policy] * (R[:, state, current_policy] + (gamma * V)))
            V = V_prime

    # used ChatGPT to determine the syntax + how to plot 
    # graphs using matplotlib library
    plt.clf()
    plt.plot(range(1, num_iterations + 1), s1_values)
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title('Value Over Time')
    
    plt.savefig(plot_filename)
    

    return V[0], V[1]


#######################################
# 8. Value Iteration for Best Policy
# 9. Output Best Policy
#######################################


# I used ChatGPT to gain a basic outline on how to code a value iteration 
# function 
# I also used ChatGPT to debug when I wasn't getting the correct optimal policy 
def value_iteration(num_iterations, plot_filename):
    V_opt = {state: 0 for state in range(num_states)}
    pi_opt = {}

    V_opt_S1_over_time = []
    
    for i in range(num_iterations):
        V_new = V_opt.copy()
        for state in range(num_states):
            action_values = []
            for action in range(num_actions):
                action_value = sum(
                    T[next_state, state, action] * (R[next_state, state, action] + gamma * V_new[next_state])
                    for next_state in range(num_states) # ChatGPT to debug to find the correct policy 
                )
                action_values.append(action_value)

            best_action = np.argmax(action_values)
            pi_opt[state] = best_action
            V_opt[state] = max(action_values)
            if (state == 0): 
                V_opt_S1_over_time.append(V_opt[state])
            

    plt.clf()
    plt.plot(range(1, num_iterations + 1), V_opt_S1_over_time)
    plt.xlabel('Iterations')
    plt.ylabel('Value of S1')
    plt.title('Value of S1 over Time')
    plt.savefig(plot_filename)  # Save the plot to the filename provided
    
    return pi_opt

def valid_actions_for_state(num_states, num_actions):
    valid_actions = {}
    for state in range(num_states):
        valid_actions[state] = []
        for action in range(num_actions):
            valid_actions[state].append(action)

    return valid_actions 


#######################################
# Main function
#######################################


def main():
    avg_reward = random_exploration()
    
    # Set first policy - S1 : A1, S2 : A3, S3 : A4, S4: no action, S5: no action
    
    policy_1 = {0 : 0, 1 : 2, 2 : 3, 3: 0, 4: 0}
    V1_1, V2_1 = evaluate_policy(policy_1, 100, "V_pi_1.png")
    

    # Set second policy - S1 : A2, S2 : A3, S3 : A4, S4: no action, S5: no action
    
    policy_2 = {0 : 1, 1 : 2, 2 : 3, 3: 0, 4: 0}
    V1_2, V2_2 = evaluate_policy(policy_2, 100, "V_pi_2.png")
    
    optimal_policy = value_iteration(100, "V_opt.png")
    print("OPTIMAL POLICY", optimal_policy)
    
    
#######################################
# DO NOT CHANGE THE FOLLOWING - AUTOGRADER JSON DUMP
#######################################

    answer = {
        "average_reward": avg_reward,
        "V1_1": V1_1,
        "V2_1": V2_1,
        "V1_2": V1_2,
        "V2_2": V2_2,
        "optimal_policy": optimal_policy,
    }

    with open("answers_MDP.json", "w") as outfile:
        json.dump(answer, outfile)


if __name__ == "__main__": 
    main()
