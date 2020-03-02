import gym
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from grid import Grid

class Agent(nn.Module):
    
    def __init__(self, epsilon=1, epsilon_decay=0.000001, epsilon_min=0.1):
        super(Agent, self).__init__()
        self.layer1 = nn.Conv2d( 8, 16, (5, 5), stride=3)
        self.layer2 = nn.Conv2d(16, 32, (3, 3), stride=2)
        self.fc     = nn.Linear(288, 4)

        nn.init.xavier_normal(self.layer1.weight)
        nn.init.xavier_normal(self.layer2.weight)
        nn.init.xavier_normal(self.fc.weight)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def action(self, observation):
        actions = observation.action_map.keys()
        valid_moves = observation.get_valid_moves(observation.positions[0])
        
        if np.random.random() < self.epsilon:
            self.decay()
            return valid_moves[np.random.randint(0, len(valid_moves))]
        
        self.decay()
        probas = self.forward(self.process_input(observation))
        best_move = None
        max_proba = 0
        for move in valid_moves:
            proba = probas[actions.index(move)]
            if proba > max_proba:
                max_proba = proba
                best_move = move
        return best_move

    def score(self, observation):
        actions = observation.action_map.keys()
        valid_moves = observation.get_valid_moves(observation.positions[0])
        probas = self.forward(self.process_input(observation))
        
        max_proba = 0
        for move in valid_moves:
            proba = probas[actions.index(move)]
            if proba > max_proba:
                max_proba = proba
        return max_proba

    def process_input(self, observation):
        x = np.stack([
            ((observation.grid & 2**(i+1)) // 2**(i+1)).reshape((1, 25, 25)) for i in range(8)
        ], axis=0).reshape((1, 8, 25, 25))
        return torch.Tensor(x)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.fc(x.view(-1))
        return x

    def update_weights(self, model):
        self.load_state_dict(model.state_dict())

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_decay
    
    def target_prediction(self, batch, target_agent):
        target_batch = numpy.zeros_like(batch)
        for j in range(len(batch)):
            batch_obs, batch_action, batch_reward, batch_next_obs = batch[j]
            if ended: 
                target_batch[j] = batch_reward
            else:
                next_target_action = target_agent.action(batch_next_observation)
                target_batch[j] = batch_reward + gamma * target_agent.score(batch_next_observation)
        return target_batch

def train(env, agent, optimizer, loss, buffer_size,
start_computing_loss = 10, update_target_agent = 100, n_episode = 5):
    buffer = deque(maxlen = buffer_size)
    loss_results = []
    n_move = 0

    for i in range(n_episode):
        ended = False
        observation = env.reset()
        while not ended:
            a = agent.action(observation)
            next_observation, reward, ended, _ = env.step(a)
            buffer.append([observation, a, reward, next_observation, ended])
            
            if (n_move % start_computing_loss == 0) and (n_move > start_computing_loss):
                shuffled_buffer = np.random.permutation(buffer)
                batch = shuffled_buffer[:batch_size]
                target_batch = target_agent.target_prediction(batch)

                batch = nn.Tensor(batch)
                target_batch = nn.Tensor(target_batch)
                batch_loss = loss(batch, target_batch)
                loss_results.append(batch_loss)
                loss.backward()
                optimizer.step()

            if n_move == update_target_agent:
                target_prediction.update_weights(agent)








    raise NotImplementedError

if __name__ == "__main__":
    grid = Grid()
    grid.create(gui_display=False)
    agent = Agent()
    input = agent.process_input(grid)
    print(agent.forward(input))

    n_epoch = 1
    learning_rate = 0.001
    buffer_size = 100

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(agent.parameters(),lr=learning_rate)





