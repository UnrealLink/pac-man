import gym
from gym import error
from gym.utils import closer
import numpy as np
import pygame

from grid import Grid
from ghost import Ghost
from gui import Gui


env_closer = closer.Closer()


class Env(object):
    r"""The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, board="board.txt", player_spawn=None, nb_ghost=4, seed=None, random_respawn=False, gui_display=False):
        """
        Create a pacman env from a txt board
        """
        self.board = board
        self.grid = Grid()
        self.random_respawn = random_respawn
        self.player_spawn = player_spawn
        self.nb_ghost = nb_ghost
        self.grid.create(board, player_spawn=player_spawn, nb_ghost=nb_ghost)
        self.free_tiles = np.argwhere(64 - (self.grid.grid & 64))
        self.base_seed = seed
        self.seed(self.base_seed)
        behaviours = Ghost.behaviour_list
        self.ghosts = [Ghost(i+1, behaviours[i]) for i in range(nb_ghost)]
        self.action_space = self.grid.get_valid_moves(self.grid.positions[0])
        self.actions = list(self.grid.action_map.keys())
        self.gui = None
        if gui_display:
            self.gui = Gui(self.grid)

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        actions = [action]
        for ghost in self.ghosts:
            actions.append(ghost.step(self.grid))
        reward, ended = self.grid.update(actions)
        self.action_space = self.grid.get_valid_moves(self.grid.positions[0])
        return self.grid, reward, ended, {}

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        if self.random_respawn:
            player_spawn = tuple(self.free_tiles[np.random.randint(0, len(self.free_tiles))])
            self.grid.reset(self.board, player_spawn=player_spawn, nb_ghost=self.nb_ghost)
        else:
            self.grid.reset(self.board, player_spawn=self.player_spawn, nb_ghost=self.nb_ghost)
        # self.seed(self.base_seed)
        if self.gui:
            self.gui = Gui(self.grid)
        return self.grid

    def render(self, mode='human'):
        """
        Renders the environment.
        """
        if self.gui:
            pygame.time.wait(250)
            self.gui.render()
        return

    def close(self):
        """
        All necessary cleanup
        """
        pass

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        """
        np.random.seed(seed)
        return

    @property
    def shape(self):
        return self.grid.grid.shape

    @property
    def unwrapped(self):
        """
        Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.close()
        # propagate exception
        return False

if __name__ == "__main__":
    env = Env()
    ended = False
    score = 0
    while not ended:
        actions = env.action_space
        action = actions[np.random.randint(0, len(actions))]
        obs, reward, ended, info = env.step(action)
        score += reward
        env.render()
    print(score)