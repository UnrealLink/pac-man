import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import utils
from pacman import Env
from grid import Grid

# Meta parameters for the RL agent
alpha = 0.1
tau = init_tau = 1
tau_inc = 0.01
gamma = 0.99
epsilon = 0.5
epsilon_decay = 0.999
verbose = True

act_to_letter = {
        0 : 'U',
        1 : 'L',
        2 : 'D',
        3 : 'R'
}

letter_to_act = {
        'U' : 0,
        'L' : 1,
        'D' : 2,
        'R' : 3
}

def coord_to_state (coords):
    return coords[0] * 25 + coords[1]

def state_to_coords (state):
    return (state//25, state%25)

def get_closest_fruits(pacman_location, safe_moves, grid):
    distances = [-1]*4
    for move in safe_moves:
        start = grid.check_position(utils.index_sum(pacman_location, grid.action_map[move]))
        flags = np.zeros(grid.grid.shape)
        queue = [(start, 0)]
        while len(queue) > 0:
            position, distance = queue.pop()
            if grid.grid[position] & 1:
                distances[letter_to_act[move]] = distance+1
                break
            moves = grid.get_valid_moves(position)
            new_positions = [utils.index_sum(position, grid.action_map[move]) for move in moves]
            for new_position in new_positions:
                new_position = grid.check_position(new_position)
                if not flags[new_position]:
                    flags[new_position] = 1
                    distance = distance + 1
                    queue = [(new_position, distance)] + queue
    return distances


def grid_to_state(grid):
    """
    Compute the state (vector of size 11) given a Grid object that represents the current observation. 
    The goal is to simplify the observation space.
    """
    state = np.zeros(11) # [s1, s2, s3, s4, s5 & 2, s5 & 1, s6, s7, s8, s9, s10]
    pacman_location = grid.positions[0]
    ghosts_location = grid.positions[1:]

    pacman_possible_moves = grid.get_valid_moves(pacman_location)

    # s1 to s4
    for move in letter_to_act.keys():
        if move not in pacman_possible_moves:
            state[letter_to_act[move]] = 1
    
    # s5
    min_distance_to_ghost_tab = -1*np.ones(4)
    dangerous_path_counter = 0
    non_dangerous_path_counter = 0
    for move in pacman_possible_moves:
        test_move = utils.index_sum(pacman_location, grid.action_map[move])
        test_move = grid.check_position(test_move)
        min_distance_to_ghost = min([grid.distances[test_move][ghost_location] for ghost_location in ghosts_location])
        min_distance_to_ghost_tab[letter_to_act[move]] = min_distance_to_ghost
        if min_distance_to_ghost < 8 :
            dangerous_path_counter += 1
        else :
            non_dangerous_path_counter +=1
    if dangerous_path_counter == len(pacman_possible_moves) or non_dangerous_path_counter == 1:
        state[4] = min_distance_to_ghost_tab.argmax() & 2
        state[5] = min_distance_to_ghost_tab.argmax() & 1
    else:
        safe_moves = []
        for move in pacman_possible_moves:
            if min_distance_to_ghost_tab[letter_to_act[move]] > 8:
                safe_moves.append(move)
        distances_to_fruits = get_closest_fruits(pacman_location, safe_moves, grid)
        state[4] = np.array([distances_to_fruits[letter_to_act[move]] for move in safe_moves]).argmin() & 2 
        state[5] = np.array([distances_to_fruits[letter_to_act[move]] for move in safe_moves]).argmin() & 1

    # s6
    for move in pacman_possible_moves:
        test_move = utils.index_sum(pacman_location, grid.action_map[move])
        test_move = grid.check_position(test_move)
        for ghost_location in ghosts_location:
            if grid.distances[test_move][ghost_location] < 8:
                state[5+letter_to_act[move]] = 1

    # since the ghosts can cut back, we only consider pacman as trapped when he will reach a ghost position whatever the move he makes
    is_trapped = np.zeros(len(pacman_possible_moves))
    for i, move in enumerate(pacman_possible_moves):
        test_move = utils.index_sum(pacman_location, grid.action_map[move])
        test_move = grid.check_position(test_move)
        for ghost_location in ghosts_location:
            if test_move == ghost_location:
                is_trapped[i] = 1
    state[9] = int(is_trapped.sum()/len(pacman_possible_moves))
    
    return state

def state_to_index(state):
    index = 0
    for i, x in enumerate(state):
        index +=  x*2**i
    return int(index)

# Act with epsilon greedy
def act_with_epsilon_greedy(index, q_table, env):
    possible_moves = env.grid.get_valid_moves(env.grid.positions[0])
    if np.random.rand() < epsilon:
        action = possible_moves[np.random.randint(len(possible_moves))]
    else:
        q_values = np.copy(q_table[index, :])
        mask = np.ones(4, dtype=bool)
        mask[np.array(list(map(lambda x : letter_to_act[x], possible_moves)))] = False
        q_values[mask] = - np.infty
        action = act_to_letter[q_values.argmax()]
    return action

# Compute Q-Learning update
def q_learning_update(q_table, index, a, r, index_prime):
    td = r + gamma * np.max(q_table[index_prime, :]) - q_table[index, letter_to_act[a]]
    return q_table[index, letter_to_act[a]] + alpha * td

# Evaluate a policy on n runs
def evaluate_policy(q_table, env, n, h):
    success_rate = 0.0 
    mean_return = 0.0

    for _ in range(n):
        discounted_return = 0.0
        observation = env.reset()
        state = grid_to_state(observation)
        index = state_to_index(state)

        for step in range(h):
            new_observation, r, done, _ = env.step(act_with_epsilon_greedy(index, q_table, env))
            discounted_return += np.power(gamma, step) * r

            if done:
                success_rate += float(r)/n
                mean_return += float(discounted_return)/n
                break

    return success_rate, mean_return

def main():

    global epsilon
    global tau

    env = Env()

    # Experimental setup
    n_episode = 10000
    print("n_episode ", n_episode)
    max_horizon = 100
    eval_steps = 10

    # Monitoring perfomance
    window = deque(maxlen=100)
    last_100 = 0

    greedy_success_rate_monitor = np.zeros([n_episode,1])
    greedy_discounted_return_monitor = np.zeros([n_episode,1])

    behaviour_success_rate_monitor = np.zeros([n_episode,1])
    behaviour_discounted_return_monitor = np.zeros([n_episode,1])

    # Init Q-table
    q_table = np.zeros((2048,4), dtype=np.float)

    env.reset()

    # Train for n_episode
    for i_episode in range(n_episode):

        # Reset a cumulative reward for this episode
        total_return = 0.0

        # Start a new episode and sample the initial state
        observation = env.reset()
        state = grid_to_state(observation)
        index = state_to_index(state)


        # First action in this episode
        a = act_with_epsilon_greedy(index, q_table, env)

        for i_step in range(max_horizon):
            
            # Act
            obsevation_prime, reward, done, _ = env.step(a)
            state_prime = grid_to_state(obsevation_prime)
            index_prime = state_to_index(state_prime)

            total_return += reward

            a_prime = act_with_epsilon_greedy(index_prime, q_table, env)

            # Update a Q value table
            q_table[index, letter_to_act[a]] = q_learning_update(q_table, index, a, reward, index_prime)

            # Transition to new state
            state = state_prime
            a = a_prime

            if done:
                window.append(reward)
                last_100 = window.count(1)

                greedy_success_rate_monitor[i_episode-1,0], greedy_discounted_return_monitor[i_episode-1,0]= evaluate_policy(q_table,env,eval_steps,max_horizon)
                behaviour_success_rate_monitor[i_episode-1,0], behaviour_discounted_return_monitor[i_episode-1,0] = evaluate_policy(q_table,env,eval_steps,max_horizon)
                if verbose:
                    print("Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tFinal_Reward: {3}\tEpsilon: {4:.3f}\tSuccess Rate: {5:.3f}\tLast_100: {6}".format(i_episode, i_step, total_return, reward, epsilon,greedy_success_rate_monitor[i_episode-1,0],last_100))
                    #print "Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tTermR: {3}\ttau: {4:.3f}".format(i_episode, i_step, total_return, r, tau)

                break


        # Schedule for epsilon
        epsilon = epsilon * epsilon_decay
        # Schedule for tau
        tau = init_tau + i_episode * tau_inc

    plt.figure(0)
    plt.plot(range(0,n_episode,10),greedy_success_rate_monitor[0::10,0])
    plt.title("Greedy policy with {0} and {1}".format("rl_algorithm", "greedy"))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")

    plt.figure(1)
    plt.plot(range(0,n_episode,10),behaviour_success_rate_monitor[0::10,0])
    plt.title("Behaviour policy with {0} and {1}".format("rl_algorithm", "explore_method"))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")
    plt.show()


if __name__ == "__main__":

    main()