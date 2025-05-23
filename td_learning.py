import random
import numpy as np
from collections import defaultdict
import pdb
import itertools

def TDLearning(episodes=1000, gamma=1, lmbda = 0, N_0 = 100):
    actions = ["hit", "stick"]
    Q = defaultdict(lambda: defaultdict(float)) # initialise state-action value function 
    E = defaultdict(lambda: defaultdict(int)) # initialise eligibility traces
    N = defaultdict(lambda: 1e-5) # number of times a state-action pair visited. a dictionary of tuples
    env = easy21_env.Easy21()
    for episode in range(episodes):
        state = env.reset() # initialise state
        action = np.random.choice(["hit","stick"],  p=[0.5, 0.5]) # initialise action
        if action not in Q[state]: Q[state][action]=0 # initialise state-action value for chosen action
        
        while not env.terminal_state:
            next_state, reward = env.step(state, action) # take action, observe reward, update state
            N[(state,action)]+=1

            # choose next action using eps-greedy policy
            eps_t = N_0/(N_0 + N[(next_state,"hit")]+ N[(next_state,"stick")]) # for eps greedy exploration. in time, eps_t becomes less and less
            is_greedy = np.random.choice([False,True],  p=[eps_t, 1-eps_t]) # with decreasing eps, exploration decreases 
            
            if is_greedy: # if greedy, choose the action producing highest value
                
                if 'hit' not in Q[next_state]: Q[next_state]['hit']=0
                if 'stick' not in Q[next_state]: Q[next_state]['stick']=0
                next_action = max((a for a in Q[next_state]), key=lambda a: Q[next_state][a])
                
            else:
                next_action = np.random.choice(["hit","stick"],  p=[0.5, 0.5])
            
            delta = reward + gamma * Q[next_state][next_action] - Q[state][action] # calculate delta
            
            if action not in E[state]: E[state][action]=0
            E[state][action] += 1
            alpha_t = 1/N[(state,action)]
            
            # TODO: complete. for all state in state space and for all action in action space
            # Q[s][a] = Q[s][a]+alpha_t*delta*E[s][a] # update q function
            # E[s][a] = gamma*lmbda*E[s][a] # update eligibility trace
            
            state, action = next_state, next_action
        
    return Q
