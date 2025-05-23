import random
import numpy as np
from collections import defaultdict
import pdb
import itertools
import easy21_env
from utils import plot_optimal_value_function, plot_optimal_policy

#### Monte Carlo Control for Easy21
# input:
# number of episodes to run, discount factor gamma (default 1), and exploration constant N_0
# returns:
# a state-action value function Q(s, a) as a nested defaultdict

def MCControl(episodes=100, gamma=1, N_0 = 100):
    actions = ["hit", "stick"]
    Q = defaultdict(lambda: defaultdict(float))# {}# defaultdict(int) # state-action value function 
    N = defaultdict(lambda: 1e-5) # number of times a state-action pair visited. a dictionary of tuples
    env = easy21_env.Easy21()
    for episode in range(episodes):
        state = env.reset()
        episode_memory = []
        episode_rewards = []

        while not env.terminal_state:
            
            eps_t = N_0/(N_0 + N[(state,"hit")]+ N[(state,"stick")]) # for eps greedy exploration. in time, eps_t becomes less and less
            is_greedy = np.random.choice([False,True],  p=[eps_t, 1-eps_t]) # with decreasing eps, exploration decreases 
            
            if is_greedy: # if greedy, choose the action producing highest value
                
                if 'hit' not in Q[state]: Q[state]['hit']=0
                if 'stick' not in Q[state]: Q[state]['stick']=0
                action = max((a for a in Q[state]), key=lambda a: Q[state][a])
                
            else:
                action = np.random.choice(["hit","stick"],  p=[0.5, 0.5])
            
            # print(f'chosen action: {action}')
            # update state
            next_state, reward = env.step(state, action)
            episode_rewards.append(reward)
            episode_memory.append((state, action))
            state = next_state
        episode_return = 0
        seen_steps = []
        for i in reversed(range(len(episode_memory))):
            state, action = episode_memory[i]
            
            if (state,action) in seen_steps:
                continue
            episode_return = gamma*episode_return+episode_rewards[i]
            seen_steps.append((state, action))
            alpha_t = 1/N[(state,action)] 
            N[(state,action)]+=1
            Q[state][action] = Q[state][action] + alpha_t*(episode_return-Q[state][action])
        
    return Q

if __name__ == "__main__":
    Q = MCControl(episodes=int(1e6), gamma=1, N_0=100)
    
    plot_optimal_value_function(Q, title="Monte Carlo Control: Value Function")
    plot_optimal_policy(Q, title="Monte Carlo Control: Optimal Policy")