import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from pacman import Env

# Meta parameters for the RL agent
alpha = 0.1
tau = init_tau = 1
tau_inc = 0.01
gamma = 0.99
epsilon = 0.5
epsilon_decay = 0.999
verbose = True

act_to_letter = {
        0 : 'D',
        1 : 'U',
        2 : 'R',
        3 : 'L'
}

letter_to_space = {
        'D' : 0,
        'U' : 1,
        'R' : 2,
        'L' : 3
}

def coord_to_state (coords):
    return coords[0] * 25 + coords[1]

def state_to_coords (state):
    return (state//25, state%25)

# Act with epsilon greedy
def act_with_epsilon_greedy(s, q, env):
    print(s)
    position = state_to_coords(s)
    possible_moves = env.grid(position)
    possible_moves = list(map(lambda x :  letter_to_act[x], possible_moves))
    q_possible = q[s, :]
    q_possible = q_possible[possible_moves]

    a = np.argmax(q_possible)
    if np.random.rand() < epsilon:
        a = np.random.randint(len(q_possible))
    #TODO a modifier quand on aura la classe Moves
    act_to_letter[possible_moves[a]]
    return a

# Compute Q-Learning update
def q_learning_update(q,s,a,r,s_prime):
    td = r + gamma * np.max(q[s_prime, :]) - q[s, letter_to_act[a]]
    return q[s,a] + alpha * td

# Evaluate a policy on n runs
def evaluate_policy(q,env,n,h):
    success_rate = 0.0
    mean_return = 0.0

    for i in range(n):
        discounted_return = 0.0
        s = env.reset()

        for step in range(h):
            s,r, done, info = env.step(act_with_epsilon_greedy(s,q))
            discounted_return += np.power(gamma,step) * r

            if done:
                success_rate += float(r)/n
                mean_return += float(discounted_return)/n
                break

    return success_rate, mean_return

def main():

    global epsilon
    global tau

    env = Env()

    # Recover State-Action space size
    #TODO a modifer quand on aura les clases action space et Moves 
    n_a = 4 #env.action_space.n
    n_s =   63*25*25 #env.observation_space.n

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
    q_table = np.zeros([n_s, n_a])

    env.reset()

    # Train for n_episode
    for i_episode in range(n_episode):

        # Reset a cumulative reward for this episode
        total_return = 0.0

        # Start a new episode and sample the initial state
        s = env.reset()
        print (s)
        s = coord_to_state(s)


        # First action in this episode
        a = act_with_epsilon_greedy(s, q_table, env)

        for i_step in range(max_horizon):

            # Act
            s_prime, r, done, info = env.step(a)

            total_return += np.power(gamma,i_step) *r

            a_prime = act_with_epsilon_greedy(s_prime, q_table, env)

            # Update a Q value table
            q_table[s, letter_to_act[a]] = q_learning_update(q_table,s,a,r,s_prime)

            # Transition to new state
            s = s_prime
            a = a_prime

            if done:
                window.append(r)
                last_100 = window.count(1)

                greedy_success_rate_monitor[i_episode-1,0], greedy_discounted_return_monitor[i_episode-1,0]= evaluate_policy(q_table,env,eval_steps,max_horizon,GREEDY)
                behaviour_success_rate_monitor[i_episode-1,0], behaviour_discounted_return_monitor[i_episode-1,0] = evaluate_policy(q_table,env,eval_steps,max_horizon,explore_method)
                if verbose:
                    print("Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tFinal_Reward: {3}\tEpsilon: {4:.3f}\tSuccess Rate: {5:.3f}\tLast_100: {6}".format(i_episode, i_step, total_return, r, epsilon,greedy_success_rate_monitor[i_episode-1,0],last_100))
                    #print "Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tTermR: {3}\ttau: {4:.3f}".format(i_episode, i_step, total_return, r, tau)

                break


        # Schedule for epsilon
        epsilon = epsilon * epsilon_decay
        # Schedule for tau
        tau = init_tau + i_episode * tau_inc

    plt.figure(0)
    plt.plot(range(0,n_episode,10),greedy_success_rate_monitor[0::10,0])
    plt.title("Greedy policy with {0} and {1}".format(rl_algorithm,explore_method))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")

    plt.figure(1)
    plt.plot(range(0,n_episode,10),behaviour_success_rate_monitor[0::10,0])
    plt.title("Behaviour policy with {0} and {1}".format(rl_algorithm,explore_method))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")
    plt.show()


if __name__ == "__main__":

    main()