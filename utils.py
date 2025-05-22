import numpy as np
import matplotlib.pyplot as plt
import pdb
import itertools

def plot_optimal_value_function(Q, title = 'Value Function'):
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
    ax.set_title(title)

    plt.show()

def plot_optimal_policy(Q, title='Optimal Policy'):
    dealers_showing = range(1, 11)
    players_sum = range(12, 22)
    policy_grid = np.zeros((len(players_sum), len(dealers_showing)))  # y by x

    for i, player in enumerate(players_sum):
        for j, dealer in enumerate(dealers_showing):
            state = (dealer, player)
            if state in Q:
                best_action = max(Q[state], key=Q[state].get)
                policy_grid[i, j] = 1 if best_action == 'stick' else 0  # 1 = stick, 0 = hit

    plt.figure(figsize=(8, 6))
    plt.imshow(policy_grid, cmap='gray', origin='lower', extent=[1, 10, 12, 21], aspect='auto')
    plt.colorbar(label='Action (0 = hit, 1 = stick)')
    plt.xlabel('Dealer Showing')
    plt.ylabel('Player Sum')
    plt.title(title)
    plt.xticks(range(1, 11))
    plt.yticks(range(12, 22))
    plt.grid(False)
    plt.show()
