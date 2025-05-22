import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pdb
import itertools

# input:
# state is a tuple (dealer's first card, player's sum)
# action is a string: "hit" or "stick"
# returns:
# a new state (dealer's card, updated player sum), or terminal if game ends
# a reward: -1, 0, or 1 depending on the outcome
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
            # play out the dealer’s cards and return the final reward and terminal state
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
        else:
            print(f'unknown action')
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
    

def MCControl(episodes=100, gamma=1, N_0 = 100):
    actions = ["hit", "stick"]
    Q = defaultdict(lambda: defaultdict(float))# {}# defaultdict(int) # state-action value function 
    N = defaultdict(lambda: 1e-5) # number of times a state-action pair visited. a dictionary of tuples
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
    ax.set_zlabel('V*(s) = max_a Q(s, a)')
    ax.set_title('Monte Carlo Control: Optimal Value Function V*(s)')

    plt.show()

def TDLearning(episodes=10000, gamma=1, N_0 = 100):
    pass
if __name__ == "__main__":
    Q = MCControl(episodes=int(1e6), gamma=1, N_0 = 100) # agent uses look up table 
    plot_optimal_value_function(Q)