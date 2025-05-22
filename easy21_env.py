from collections import defaultdict
import pdb
import itertools
import random
import numpy as np
#### Easy21 environment
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
    