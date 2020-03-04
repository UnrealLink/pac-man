import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle
import pygame

import utils
from pacman import Env
from grid import Grid



class Agent(object):

    def __init__(self, q_table, epsilon=0.5, espilon_decay=0.999, gamma=0.99, alpha=0.1):
        self.q_table = q_table
        self.epsilon = epsilon
        self.epsilon_decay = espilon_decay
        self.gamma = gamma
        self.alpha = alpha

    def act_with_epsilon_greedy(self, index, env):
        possible_moves = env.grid.get_valid_moves(env.grid.positions[0])
        if np.random.rand() < epsilon:
            action = possible_moves[np.random.randint(len(possible_moves))]
        else:
            q_values = np.copy(self.q_table[index, :])
            mask = np.ones(4, dtype=bool)
            mask[np.array(list(map(lambda x : letter_to_act[x], possible_moves)))] = False
            q_values[mask] = - np.infty

            best_moves = np.where(q_values == q_values.max())[0]
            
            # in case they are several best moves, the agent chooses randomly
            p = 1/len(best_moves)
            proba = np.array([p] * len(best_moves))
            best_move = np.random.choice(best_moves, p=proba)

            action = act_to_letter[best_move]
        return action
    
    def update_q_table(self, index, next_index, action, reward):
        td = reward + self.gamma * np.max(self.q_table[next_index, :]) - self.q_table[index, letter_to_act[action]]
        self.q_table[index, letter_to_act[action]] = self.q_table[index, letter_to_act[action]] + self.alpha * td
        return 

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay
        return

    def evaluate_policy(self, nb_games=50, max_steps=100):
        
        mean_score = 0.
        mean_steps = 0.

        for _ in range(nb_games):
            observation = env.reset()
            state = grid_to_state(observation)
            index = state_to_index(state)
            cumulated_reward = 0

            for step_number in range(max_steps):
                new_observation, reward, done, _ = env.step(act_with_epsilon_greedy(self, index, env))
                cumulated_reward += reward
                if done:
                    mean_score += cumulated_reward / nb_games
                    mean_steps += step_number / nb_games
                    break

        return mean_score, mean_steps

    def grid_to_state(self, grid):
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
        if non_dangerous_path_counter == 0 or non_dangerous_path_counter == 1:
            # print('check1')
            # print(act_to_letter[min_distance_to_ghost_tab.argmax()])
            state[4] = min_distance_to_ghost_tab.argmax() & 2
            state[5] = min_distance_to_ghost_tab.argmax() & 1
        else:
            safe_moves = []
            for move in pacman_possible_moves:
                if min_distance_to_ghost_tab[letter_to_act[move]] >= 8:
                    safe_moves.append(move)
            distances_to_fruits = self.get_closest_fruits(safe_moves, grid)
            # print('check2')
            # print(distances_to_fruits)
            for i in range(len(distances_to_fruits)):
                if act_to_letter[i] not in safe_moves or distances_to_fruits[i] == -1 :
                    distances_to_fruits[i] = np.infty
            # print(distances_to_fruits)
            # print(act_to_letter[np.array(distances_to_fruits).argmin()])
            state[4] = np.array(distances_to_fruits).argmin() & 2
            state[5] = np.array(distances_to_fruits).argmin() & 1

        # s6
        # print('S6...')
        for move in pacman_possible_moves:
            # print(move)
            test_move = utils.index_sum(pacman_location, grid.action_map[move])
            test_move = grid.check_position(test_move)
            for ghost_location in ghosts_location:
                # print(grid.distances[test_move][ghost_location])
                if grid.distances[test_move][ghost_location] < 8:
                    state[6+letter_to_act[move]] = 1
        # print('...S6')
        
        # since the ghosts can cut back, we only consider pacman as trapped when he will reach a ghost position whatever the move he makes
        is_trapped = np.zeros(len(pacman_possible_moves))
        for i, move in enumerate(pacman_possible_moves):
            test_move = utils.index_sum(pacman_location, grid.action_map[move])
            test_move = grid.check_position(test_move)
            for ghost_location in ghosts_location:
                if test_move == ghost_location:
                    is_trapped[i] = 1
        state[10] = int(is_trapped.sum()/len(pacman_possible_moves))
        
        return state
    
    def state_to_index(self, state):
        index = 0
        for i, x in enumerate(state):
            index +=  int(x)*2**i
        return index

    def get_closest_fruits(self, grid, safe_moves):
        pacman_location = grid.positions[0]
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



def train(agent, board='board.txt', epoch=10000, max_horizon=100, evaluation_frequency=1000, number_games_for_evaluation=50, saving_frequency=1000, saving=True, verbose=True):

    mean_reward_evolution = []

    env = Env(board=board, random_respawn=True)
    
    for game in range(epoch + 1):
        total_reward_of_the_game = 0
        observation = env.reset()
        state = agent.grid_to_state(observation)
        index = agent.state_to_index(state)
        action = agent.act_with_epsilon_greedy(index, env)

        for _ in range(max_horizon + 1):
            next_obsevation, reward, done, _ = env.step(action)
            total_reward_of_the_game += reward
            next_state = agent.grid_to_state(next_obsevation)
            next_index = agent.state_to_index(next_state)
            next_action = agent.act_with_epsilon_greedy(next_index, env)

            agent.update_q_table(index, next_index, action, reward)

            state = next_state
            index = next_index
            action = next_action

            if done:
                if game % evaluation_frequency == 0:
                    mean_reward, mean_steps = agent.evaluate_policy(nb_games=number_games_for_evaluation)
                    mean_reward_evolution.append(mean_reward)
                    if verbose:
                        print(f"Game : {game} \t Reward of the game : {total_reward_of_the_game} \t")
                        print(f"Mean score of the agent on {number_games_for_evaluation} \t Mean reward: {mean_reward} \t Mean steps before end: {mean_steps}")
                if saving and game % saving_frequency == 0:
                    with open(f'../q_tables/q_table_{game}.pickle', 'wb') as q_table_file:
                        pickle.dump(agent.q_table, q_table_file)

        agent.update_epsilon()

    plt.figure()
    plt.plot(range(epoch//evaluation_frequency), mean_reward_evolution)
    plt.title(f"Greedy policy mean reward evolution on {number_games_for_evaluation} game(s)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean reward")
    plt.show()



def evaluate_model(agent, nb_games, verbose=True):

    pass 


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



def evaluate_model(board, path):
    env = Env(board=board, random_respawn=False, gui_display=True)
    ended = False
    with open(path, 'rb') as q_table_file:
        q_table = pickle.load(q_table_file)
    observation = env.grid
    print(q_table)
    while not ended:
        state = grid_to_state(observation)
        # print(state)
        index = state_to_index(state)
        # print(index)
        action = act_with_epsilon_greedy(index, q_table, env, 0)
        # print(action)
        # print(act_to_letter[state[4]*2+state[5]])
        observation, reward, ended, _ = env.step(action)
        env.gui.score += reward
        env.render()
        pygame.time.wait(250)

def train(board):
    

    env = Env(board=board, random_respawn=True)

    # Experimental setup
    n_episode = 10001
    print("n_episode ", n_episode)
    max_horizon = 100
    eval_steps = 10

    # Monitoring perfomance
    window = deque(maxlen=100)
    last_100 = 0

    # Init Q-table
    q_table = np.zeros((2048,4), dtype=np.float)
    # for index in range(2048):
    #     state = np.binary_repr(index, width=11)
    #     prefered_action = 2*int(state[4])+int(state[5])
    #     q_table[index, prefered_action] = 0.5

    env.reset()

    # Train for n_episode
    for i_episode in range(n_episode):

        # Reset a cumulative reward for this episode
        total_return = 0.0

        # Start a new episode and sample the initial state
        observation = env.reset()
        # print('...state')
        state = grid_to_state(observation)
        # print('...state')
        index = state_to_index(state)

        # print(f"State : {state}")


        # First action in this episode
        a = act_with_epsilon_greedy(index, q_table, env, epsilon)

        for i_step in range(max_horizon):

            # Act
            obsevation_prime, reward, done, _ = env.step(a)
            # print('state_prime...')
            state_prime = grid_to_state(obsevation_prime)
            # print('...state_prime')
            index_prime = state_to_index(state_prime)

            total_return += reward

            a_prime = act_with_epsilon_greedy(index_prime, q_table, env, epsilon)

            # Update a Q value table
            q_table[index, letter_to_act[a]] = q_learning_update(q_table, index, a, reward, index_prime)

            # Transition to new state
            state = state_prime
            a = a_prime

            if done:
                window.append(total_return)
                last_100 = window.count(1)

                greedy_success_rate_monitor[i_episode-1,0], greedy_discounted_return_monitor[i_episode-1,0]= evaluate_policy(q_table,env,eval_steps,max_horizon)
                if verbose and i_episode % 25 == 0:
                    print("Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tFinal_Reward: {3}\tEpsilon: {4:.3f}\tSuccess Rate: {5:.3f}\tLast_100: {6}".format(i_episode, i_step, total_return, reward, epsilon,greedy_success_rate_monitor[i_episode-1,0],last_100))
                break

        if i_episode % 1000 == 0:
            with open(f'../q_tables/q_table_{i_episode}.pickle', 'wb') as q_table_file:
                pickle.dump(q_table, q_table_file)
            np.savetxt(f'../q_tables/q_table_{i_episode}.csv', q_table, delimiter=',')
        # Schedule for epsilon
        epsilon = decrese_epsilon(i_episode, epsilon)


    plt.figure(0)
    plt.plot(range(0,n_episode,10),greedy_success_rate_monitor[0::10,0])
    plt.title("Greedy policy with {0} and {1}".format("rl_algorithm", "greedy"))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")
    plt.show()

def main(board, eval=False):
    if eval: 
        for i in range (10, 11):
            print(500*i)
            for tests in range(10):
                evaluate_model(board, f'../q_tables/q_table_{10000}.pickle')
    else: 
        train(board)

if __name__ == "__main__":

    main('board.txt', eval=True)