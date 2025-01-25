import random
import numpy as np
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
        return (self.dealers_card, self.players_card), self.terminal_state
    
    def step(self, state, action):
        dealers_sum = state[0][0]
        players_sum = state[0][1]
        reward = 0
        if action == 'stick':
            # TODO: play out the dealer’s cards and return the final reward and terminal state
            dealers_sum, reward = self.playout(dealers_sum, players_sum)
            self.terminal_state = True
        elif action == 'hit':
            players_sum += self.sample_card()
            if players_sum > 21:
                reward = -1
                self.terminal_state = True
        else:
            print(f'unknowm action')
            return
        next_state = (dealers_sum, players_sum)
        return next_state, reward, self.terminal_state
    
    # choose between -1 (p=1/3) and +1 (p=2/3) 
    def sample_card(self):
        color = np.random.choice([-1,1],  p=[1/3, 2/3])
        number = random.randint(1, 10)
        return int(color*number)
    
    def playout(self, dealers_sum, players_sum):
        while dealers_sum < 17:
                print(dealers_sum)
                dealers_sum += self.sample_card()
        if dealers_sum >21 or dealers_sum < players_sum:
            # TODO: update next state
            reward = 1
        elif dealers_sum == players_sum: # draw
            reward = 0
        elif dealers_sum == 21 or dealers_sum > players_sum:
            # TODO: update next state
            reward = -1
        else:
            print(f'Unknown state occurred with dealers sum:{dealers_sum}, players sum:{players_sum}')
        return dealers_sum, reward
""" 
Apply Monte-Carlo control to Easy21. Initialise the value function to zero. Use
a time-varying scalar step-size of alphat = 1/N(st, at) and an epsilon-greedy exploration
strategy with t = N0/(N0 + N(st)), where N0 = 100 is a constant, N(s) is
the number of times that state s has been visited, and N(s, a) is the number
of times that action a has been selected from state s. Feel free to choose an
alternative value for N0, if it helps producing better results. Plot the optimal
value function V
∗
(s) = maxa Q∗
(s, a) using similar axes to the following figure
taken from Sutton and Barto’s Blackjack example.
"""
def MCControl(episodes=100, gamma=1, N_0 = 100):
    actions = ["hit", "stick"]
    Q = {} # state-action value function 
    N = {} # number of times a state-action pair visited
    returns = {} # returns per state-action pair
    env = Easy21()
    for episode in range(episodes):
        state, terminal_state = env.reset()
        episode_memory = []
        episode_return = 0
        while not terminal_state:
            
            eps_t = N_0/(N_0 + N[(state,"hit")]+ N[(state,"stick")])
            # TODO: choose action
            is_greedy = np.random.choice([False,True],  p=[eps_t, 1-eps_t])
            if is_greedy:
                action = max((action for (s, a) in Q if s == state),
                                key=lambda action: Q[(state, action)])
            else:
                action = np.random.choice(["hit","stick"],  p=[0.5, 0.5])
            N[(state,action)]+=1
            
            # update state
            state, reward, terminal_state = env.step(state, action)
            episode_return += reward
            alpha_t = 1/N[(state,action)] # update at the end of episode
            Q[(state, action)] = Q[(state, action)] + alpha_t*(episode_return-Q[(state, action)])
    return Q

Q = MCControl(episodes=100, gamma=1, N_0 = 100)
