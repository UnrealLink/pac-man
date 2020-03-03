import gym
import numpy as np
from collections import deque
from tqdm import tqdm
import copy
import time
import pygame

import torch
import torch.nn as nn
import torch.nn.functional as F

from grid import Grid
from pacman import Env

class Agent(nn.Module):
    
    def __init__(self, training=False, epsilon=1, epsilon_decay=0.000001, epsilon_min=0.1):
        super(Agent, self).__init__()
        self.layer1 = nn.Conv2d( 8, 16, (5, 5), stride=3)
        self.layer2 = nn.Conv2d(16, 32, (3, 3), stride=2)
        self.fc     = nn.Linear(288, 4)

        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.xavier_normal_(self.fc.weight)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.training = training

    def action(self, observation):
        actions = list(observation.action_map.keys())
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

    def action_with_score(self, observation):
        actions = list(observation.action_map.keys())
        valid_moves = observation.get_valid_moves(observation.positions[0])
        probas = self.forward(self.process_input(observation))
        
        if np.random.random() < self.epsilon:
            self.decay()
            index = np.random.randint(0, len(valid_moves))
            return (valid_moves[index], probas[index])
        
        self.decay()
        best_move = None
        max_proba = -np.infty
        for move in valid_moves:
            proba = probas[actions.index(move)]
            if proba > max_proba:
                max_proba = proba
                best_move = move
        return best_move, max_proba

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
    
    def target_prediction(self, batch, target_agent, gamma):
        target_score = np.zeros(len(batch))
        for j in range(len(batch)):
            obs, action, reward, next_obs, ended, score = batch[j]
            if ended: 
                target_score[j] = reward
            else:
                target_score[j] = reward + gamma * target_agent.action_with_score(next_obs)[1]
        return target_score

def train(env, agent, optimizer, loss, buffer_size=100, batch_size=32, gamma=0.95, n_episode = 1000,
          start_computing_loss = 10, update_target_agent = 10000, save_model=500, name="model"):
    target_agent = Agent(epsilon=0)
    target_agent.update_weights(agent)
    agent.training = True
    buffer = deque(maxlen = buffer_size)
    loss_results = []
    n_move = 0
    max_fruits = 257
    all_scores = []

    for episode in tqdm(range(n_episode)):
        ended = False
        observation = Grid.copy(env.reset())
        while not ended:
            action, score = agent.action_with_score(observation)
            next_obs, reward, ended, _ = env.step(action)
            if score:
                buffer.append([observation, action, reward, next_obs, ended, score])

            if (n_move % start_computing_loss == 0) and (n_move >= start_computing_loss):
                shuffled_buffer = np.random.permutation(buffer)
                batch = shuffled_buffer[:batch_size]
                target_score = target_agent.target_prediction(batch, target_agent, gamma)

                batch_score = [x[-1] for x in batch]
                target_score = torch.Tensor(target_score)
                for i, scores in enumerate(zip(batch_score, target_score)):
                    s, target_s = scores
                    partial_loss = loss(s, target_s)
                    if i == len(batch_score):
                        partial_loss.backward()
                    else:
                        partial_loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

            if (n_move % update_target_agent == 0) and (n_move >= update_target_agent):
                target_agent.update_weights(agent)

            observation = Grid.copy(next_obs)
            n_move += 1

        if ((episode+1) % save_model == 0) and ((episode+1) >= save_model):
            torch.save(agent.state_dict(), f"models/{name}_{episode+1}.pth")

        all_scores.append(max_fruits - observation.nb_fruits)

    with open(f"info/{name}_scores.txt", 'w') as file:
        file.writelines(["%s\n" % item  for item in all_scores])

def evaluate_model(path, player_spawn=None):
    env = Env(gui_display=True, player_spawn=player_spawn)
    env.seed(42)
    agent = Agent(epsilon=0)
    agent.load_state_dict(torch.load(path))
    ended = False
    observation = env.grid
    while not ended:
        pygame.time.wait(250)
        action = agent.action(observation)
        observation, reward, ended, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    env = Env()
    env.seed(42)
    agent = Agent()

    learning_rate = 0.0001

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(agent.parameters(),lr=learning_rate)

    train(env, agent, optimizer, loss, n_episode=10000, save_model=2000, name="run8")
    # evaluate_model('models/run7_3000.pth')



