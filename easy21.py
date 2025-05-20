import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pdb
import itertools
# inputs: 
# a tuple of ints for state: dealer’s first card 1-10 and the player’s sum 1-21
# a string for action: hit or stick
# returns:
# a sample of the next state s, may be terminal if the game is finished
# reward r
class Easy21():
    
    def reset(self):
        self.dealers_card = random.randint(1, 10)
        self.players_card = random.randint(1, 10)
        self.terminal_state = False
        # print(f'dealers card: {self.dealers_card}, players card: {self.players_card}')
        return self.dealers_card, self.players_card # (self.dealers_card, self.players_card), self.terminal_state
    
    def step(self, state, action):
        dealers_sum = state[0]
        players_sum = state[1]
        reward = 0
        
        if action == 'stick':
            # TODO: play out the dealer’s cards and return the final reward and terminal state
            while dealers_sum < 17:
                dealers_sum += self.sample_card()
            self.terminal_state = True
            if dealers_sum>21 or players_sum>dealers_sum:
                reward = 1
            elif dealers_sum > players_sum:
                reward = -1
            elif dealers_sum == players_sum:
                reward = 0
            else:
                print(f"missing condition!")

        elif action == 'hit':
            players_sum += self.sample_card()
            if players_sum > 21:
                reward = -1
                self.terminal_state = True
            if players_sum == 21:
                reward = 1
                self.terminal_state = True
        else:
            print(f'unknowm action')
            return
        # print(action, players_sum)
        next_state = (self.dealers_card, players_sum)
        return next_state, reward
    
    # choose between -1 (p=1/3) and +1 (p=2/3) 
    def sample_card(self):
        color = np.random.choice([-1,1],  p=[1/3, 2/3])
        # print(f'color:{color}')
        number = random.randint(1, 10)
        return int(color*number)
    
   
""" 
Apply Monte-Carlo control to Easy21. Initialise the value function to zero. Use
a time-varying scalar step-size of alphat = 1/N(st, at) and an epsilon-greedy exploration
strategy with t = N0/(N0 + N(st)), where N0 = 100 is a constant, N(s) is
the number of times that state s has been visited, and N(s, a) is the number
of times that action a has been selected from state s. Feel free to choose an
alternative value for N0, if it helps producing better results. Plot the optimal
value function V
(s) = maxa 
(s, a) using similar axes to the following figure
taken from Sutton and Barto’s Blackjack example.
"""
def MCControl(episodes=100, gamma=1, N_0 = 100):
    actions = ["hit", "stick"]
    Q = defaultdict(lambda: defaultdict(float))# {}# defaultdict(int) # state-action value function 
    N = defaultdict(lambda: 1e-5) # number of times a state-action pair visited. a dictionary of tuples
    returns = {} # returns per state-action pair
    env = Easy21()
    for episode in range(episodes):
        state = env.reset()
        episode_memory = []
        episode_rewards = []

        while not env.terminal_state:
            
            eps_t = N_0/(N_0 + N[(state,"hit")]+ N[(state,"stick")]) # for eps greedy exploration. in time, eps_t becomes less and less
            # TODO: choose action
            is_greedy = np.random.choice([False,True],  p=[eps_t, 1-eps_t]) # with decreasing eps, exploration decreases 
            if is_greedy: # if greedy, choose the action producing highest value
                # print(f'it is greedy now in episode: {episode}!')
                
                if 'hit' not in Q[state]: Q[state]['hit']==0
                if 'stick' not in Q[state]: Q[state]['stick']==0
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
        for i in range(len(episode_memory)):
            state, action = episode_memory[i]
            if (state,action) in seen_steps:
                continue
            seen_steps.append((state, action))
            episode_return += episode_rewards[i]
            alpha_t = 1/N[(state,action)] # update at the end of episode
            N[(state,action)]+=1
            Q[state][action] = Q[state][action] + alpha_t*(episode_return-Q[state][action])
        
    return Q

def plot_optimal_value_function(Q):
    dealers_showing = [i for i in range(1,11)]
    players_sum = [i for i in range(12,21)]
    
    X = []
    Y = []
    Z = []

    for dealer, player in itertools.product(dealers_showing, players_sum):
        best_value = max(Q[(dealer, player)].values())
        X.append(dealer)
        Y.append(player)
        Z.append(best_value)
    
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_zlabel('Best Value')
    ax.set_title('Optimal Value Function (Best Action Value per State)')

    plt.show()

def TDLearning(episodes=10000, gamma=1, N_0 = 100):
    pass
if __name__ == "__main__":
    Q = MCControl(episodes=int(1e5), gamma=1, N_0 = 100) # agent uses look up table 
    plot_optimal_value_function(Q)