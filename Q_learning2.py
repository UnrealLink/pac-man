import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import utils
from pacman import Env
from grid import Grid
from gui import Gui

# Meta parameters for the RL agent
alpha = 0.1
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
def act_with_epsilon_greedy(state, w, env):
    possible_moves = env.grid.get_valid_moves(env.grid.positions[0])
    if np.random.rand() < epsilon:
        action = possible_moves[np.random.randint(len(possible_moves))]
    else:
        q_values = [compute_q(state, move, w) for move in ['U','L','D','R']]
        mask = np.ones(4, dtype=bool)
        mask[np.array(list(map(lambda x : letter_to_act[x], possible_moves)))] = False
        for i,poss_move in enumerate(mask):
            if poss_move:
                q_values[i] = - np.infty
        action = act_to_letter[np.argmax(q_values)]
    return action

def compute_q (state, a, w):
    a = letter_to_act[a] + 1 # the "+1" is here to prevent us from having letter to act[a]=0 which would make the action to have no influence on Q
    biais_state_a = list(state)
    biais_state_a.insert(0,1)
    biais_state_a.append(a)
    return sum([x * y for x, y in zip(biais_state_a, w)])

def compute_q_max(state, w, env):
    possible_moves = env.grid.get_valid_moves(env.grid.positions[0])
    q_values = [compute_q(state, move, w) for move in ['U','L','D','R']]
    mask = np.ones(4, dtype=bool)
    mask[np.array(list(map(lambda x : letter_to_act[x], possible_moves)))] = False
    for i,poss_move in enumerate(mask):
        if poss_move:
            q_values[i] = - np.infty
    return act_to_letter[np.argmax(q_values)], max(q_values)

def main(test_mode=False, n_episode=100, w=[0]*13, max_horizon=100):

    global epsilon

    env = Env(gui_display=test_mode)

    # Experimental setup
    ###n_episode = 1000
    ###print("n_episode ", n_episode)
    ###max_horizon = 100

    # Init weights
    ###w = [0]*13

    env.reset()

    # Train for n_episode
    for i_episode in range(n_episode):

        # Reset a cumulative reward for this episode
        total_return = 0.0

        # Start a new episode and sample the initial state
        observation = env.reset()
        state = grid_to_state(observation)
        # First action in this episode
        a = act_with_epsilon_greedy(state, w, env)

        ######
        
        q_s_a = compute_q(state, a, w)

        #######

        for _ in range(max_horizon):
            
            # Act
            obsevation_prime, reward, done, _ = env.step(a)
            state_prime = grid_to_state(obsevation_prime)

            ######

            biais_state_a = list(state)
            biais_state_a.append(letter_to_act[a] + 1)
            biais_state_a.insert(0,1)
            a_prime, q_prim = compute_q_max(state_prime, w, env)
            w = np.array(w)
            biais_state_a = np.array(biais_state_a)
            if not test_mode:
                w = w + alpha*(reward + gamma*q_prim - q_s_a)*biais_state_a
            else:
                env.render()
            #######

            total_return += reward

            # Transition to new state
            state = state_prime
            a = a_prime

            if done:
                print(i_episode)
                print(total_return)
                break


        # Schedule for epsilon
        epsilon = epsilon * epsilon_decay
    return w


if __name__ == "__main__":

    w_ = main()
    print("train finished")
    w_ = main(True, n_episode=1, w=w_, max_horizon=10)
