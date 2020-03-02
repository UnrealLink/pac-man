import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    
    def __init__(self):
        super(Agent, self).__init__()
        self.layer1 = nn.Conv2d( 8, 16, (5, 5), stride=3)
        self.layer2 = nn.Conv2d(16, 32, (3, 3), stride=2)
        self.fc     = nn.Linear(288, 4)


    def action(self, observation):
        raise NotImplementedError

    def process_input(self, observation):
        raise NotImplementedError

    def forward(self, input):
        x = input
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.fc(x.view(-1))
        x = torch.sigmoid(x)
        return x

    def update_weights(self, weights):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError


if __name__ == "__main__":
    agent = Agent()
    input = torch.Tensor(np.zeros((1, 8, 25, 25)))
    print(agent.forward(input))
