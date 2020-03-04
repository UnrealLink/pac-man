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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Agent(nn.Module):
    
    def __init__(self, input_size, epsilon=1, epsilon_decay=0.000001, epsilon_min=0.1):
        super(Agent, self).__init__()
        self.input_size = input_size
        
        # Compute size of last conv layer
        size = ((input_size[0] - 5) // 2 + 1) - 2

        # Define net layers
        self.layer1 = nn.Conv2d( 1, 16, (5, 5), stride=2)
        self.layer2 = nn.Conv2d(16, 32, (3, 3), stride=1)
        self.fc     = nn.Linear(size*32, 4)

        # Init layers
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        # Parameters for epsilon greedy policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.fc(x.view(x.shape[0], -1))
        return x

    def process_input(self, observations):
        """
        Transform a batch of grids into a batch of tensors
        Input should be an iterable of grids
        """
        shape = (1, self.input_size[0], self.input_size[1])
        x = np.array([observation.grid.reshape(shape) for observation in observations], dtype=np.uint8)
        # pylint: disable=not-callable
        return torch.tensor(x, dtype=torch.float32, device=device)

    def action(self, observation):
        """
        Select an action according to observation
        """

        actions = list(observation.action_map.keys())
        valid_moves = observation.get_valid_moves(observation.positions[0])
        
        # Reduces epsilon
        self.decay()

        # Select a random move with a probability of epsilon
        if np.random.random() < self.epsilon:
            return valid_moves[np.random.randint(0, len(valid_moves))]
        
        # Otherwise uses the net output
        with torch.no_grad():
            scores = self.forward(self.process_input([observation])).numpy()[0]
        for i, action in enumerate(actions):
            if not (action in valid_moves):
                scores[i] = - np.infty
        return actions[np.argmax(scores)]

    def decay(self):
        """
        Reduces epsilon
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_decay
    
    def optimize(self, batch, target_agent, optimizer, env):
        """
        Optimize the agent on a batch according to target_agent's predictions
        """
        GAMMA = 0.95

        state_batch      = batch[:, 0]
        action_batch     = batch[:, 1]
        reward_batch     = batch[:, 2]
        next_state_batch = batch[:, 3]

        # Compute the score of each (state, action) according to our net
        actions = env.actions
        # pylint: disable=not-callable
        indices = torch.tensor([actions.index(action) for action in action_batch], dtype=torch.int64, device=device).reshape((len(batch), 1))
        state_action_scores = self.forward(self.process_input(state_batch)).gather(1, indices)

        # Compute the best score of target_agent for the next step of non final moves
        next_state_scores = torch.zeros(len(batch), device=device)
        non_final_next_states_indices = np.argwhere(next_state_batch != None).reshape(-1)
        non_final_next_states = next_state_batch[non_final_next_states_indices]
        next_state_scores[non_final_next_states_indices] = target_agent(self.process_input(non_final_next_states)).max(1)[0].detach() # we don't need the gradient

        # Compute expected Q values
        reward_batch = reward_batch.astype(np.float32)
        expected_state_action_scores = torch.tensor(reward_batch, device=device) + (next_state_scores * GAMMA)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_scores, expected_state_action_scores.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # Clipping gradient
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def train_agent(self, env, num_episodes=1000, save_model=500, name="model"):
        """
        Train the model on env
        """

        # Create memory
        BUFFER_SIZE = 100
        BATCH_SIZE  = 32
        memory = deque(maxlen=BUFFER_SIZE)

        # Optimizer
        optimizer = torch.optim.RMSprop(self.parameters())

        # Create target net to compute loss
        target_agent = Agent(self.input_size)
        target_agent.load_state_dict(self.state_dict())
        target_agent.eval()
        UPDATE_TARGET_AGENT = 100

        scores = []

        for episode in tqdm(range(1, num_episodes+1)):
            # Initialize the environment and state
            current_state = Grid.copy(env.reset())
            done = False
            score = 0

            # Run the game
            while not done:
                # Select and perform an action
                action = self.action(current_state)
                observation, reward, done, _ = env.step(action)                
                next_state = None if done else Grid.copy(observation)
                score += reward

                # Add transition to memory
                memory.append([current_state, action, reward, next_state])

                # Update state
                current_state = Grid.copy(observation)

                # If memory is filled enough, optimize model with a transition batch
                if len(memory) >= BATCH_SIZE:
                    shuffled_memory = np.random.permutation(memory)
                    batch = shuffled_memory[:BATCH_SIZE]
                    self.optimize(batch, target_agent, optimizer, env)

            # Update target net weights
            if episode % UPDATE_TARGET_AGENT == 0:
                target_agent.load_state_dict(self.state_dict())
            
            # Saves weights to file
            if episode % save_model == 0:
                torch.save(self.state_dict(), f"models/{name}_{episode}.pth")
            
            scores.append(score)

        # Saves score to file
        with open(f"info/{name}_scores.txt", 'w') as file:
            file.writelines(["%s\n" % item  for item in scores])


def evaluate_model(path, board):
    env = Env(board, gui_display=True, nb_ghost=1)
    env.seed(42)
    agent = Agent(env.grid.grid.shape, epsilon=0)
    agent.load_state_dict(torch.load(path))
    ended = False
    observation = env.grid
    while not ended:
        pygame.time.wait(500)
        action = agent.action(observation)
        observation, reward, ended, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    # env = Env("board2.txt", nb_ghost=1)
    # env.seed(34)
    # agent = Agent(env.shape)
    # agent.train_agent(env, num_episodes=10000, save_model=5000)

    evaluate_model("models/model_10000.pth", "board2.txt")





    