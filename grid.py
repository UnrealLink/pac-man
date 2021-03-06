import numpy as np
from hashlib import sha1
from copy import copy

from utils import index_sum, InvalidIndex


np.set_printoptions(linewidth=120)

class Grid(object):
    """ 
    Implements the game board. 25x25 grid with:
        0  -> empty
        1  -> fruit
        2  -> player
        4  -> ghost 1
        8  -> ghost 2
        16 -> ghost 3
        32 -> ghost 4
        64 -> wall
    and sum of above numbers if some of them are overlapping.
    """

    action_map = {
        'D': (1, 0),
        'U': (-1, 0),
        'R': (0, 1),
        'L': (0, -1)
    }

    def __init__(self):
        pass

    def create(self, board="board.txt", player_spawn=None, ghost_spawn=None, nb_ghost=4):
        """
        Initialize a grid with the parameters contained in board
        """
        with open(board, 'r') as board_file:
            self.player_spawn = tuple(map(lambda x : int(x), board_file.readline().split()))
            self.ghost_spawn  = tuple(map(lambda x : int(x), board_file.readline().split()))
            self.grid = np.array([line.split() for line in board_file.readlines()], dtype=np.int8)
        if player_spawn:
            self.player_spawn = player_spawn
        if ghost_spawn: 
            self.ghost_spawn = ghost_spawn
        self.nb_ghost = nb_ghost
        self.old_positions = [(0,0)]*(1+self.nb_ghost)
        self.positions = [self.player_spawn] + [self.ghost_spawn]*self.nb_ghost   # 0: player, 1-self.nb_ghost: ghosts
        for i, position in enumerate(self.positions):
            self.grid[position] = self.grid[position] + 2**(i+1)
        self.nb_fruits = np.sum(self.grid.reshape(-1) & 1)
        self.last_point_taken = 0
        self.distances = {}
        self.compute_distances()

    @classmethod
    def copy(cls, grid):
        """
        Copy a grid into a new instance
        """
        new_grid = Grid()
        new_grid.grid = np.copy(grid.grid)
        new_grid.player_spawn = grid.player_spawn
        new_grid.ghost_spawn  = grid.ghost_spawn
        new_grid.nb_ghost = grid.nb_ghost
        new_grid.positions = list(grid.positions)
        new_grid.old_positions = list(grid.old_positions)
        new_grid.nb_fruits = grid.nb_fruits
        new_grid.distances = grid.distances
        return new_grid

    def reset(self, board="board.txt", player_spawn=None, ghost_spawn=None, nb_ghost=4):
        """
        Reset a grid with the parameters contained in board (does not compute the distances again)
        """
        with open(board, 'r') as board_file:
            self.player_spawn = tuple(map(lambda x : int(x), board_file.readline().split()))
            self.ghost_spawn  = tuple(map(lambda x : int(x), board_file.readline().split()))
            self.grid = np.array([line.split() for line in board_file.readlines()], dtype=np.int8)
        if player_spawn:
            self.player_spawn = player_spawn
        if ghost_spawn: 
            self.player_spawn = ghost_spawn
        self.positions = [self.player_spawn] + [self.ghost_spawn]*nb_ghost   # 0: player, 1-4: ghosts 
        for i, position in enumerate(self.positions):
            self.grid[position] = self.grid[position] + 2**(i+1)
        self.nb_fruits = np.sum(self.grid.reshape(-1) & 1)
        self.last_point_taken = 0

    def update(self, actions):
        """ 
        Update the grid according to agent and ghosts' actions.
        actions is a char list of size 5 containing the action ('U', 'D', 'R', 'L')
        return (reward, ended) tuple.
        """
        self.last_point_taken += 1
        self.old_positions = copy(self.positions)
        for i, action in enumerate(actions):
            self.grid[self.positions[i]] = self.grid[self.positions[i]] - 2**(i+1)
            self.positions[i] = self.check_position(index_sum(self.positions[i], self.action_map[action]))
            self.grid[self.positions[i]] = self.grid[self.positions[i]] + 2**(i+1)
        reward = self.compute_reward()
        ended  = self.check_ending()
        return (reward, ended)

    def check_position(self, position):
        """
        Check if position is valid
        """
        position = (position[0]%self.grid.shape[0], position[1]%self.grid.shape[1])
        if self.grid[position] == 64:
            raise InvalidIndex("Invalid position")
        return position

    def compute_reward(self):
        """
        Check if the player got a reward for moving to his position
        Remove the fruit if needed
        """
        # See if a ghost has eaten the agent
        for i, position in enumerate(self.positions[1:]):
            if self.positions[0] == position:
                return -50
        for i, position in enumerate(self.old_positions[1:]):
            if self.positions[0] == position and self.old_positions[0] == self.positions[i+1]:
                return -50

        # See if the agent has eaten a fruit
        if self.grid[self.positions[0]] & 1:
            self.grid[self.positions[0]] = self.grid[self.positions[0]] - 1
            self.nb_fruits -= 1
            
            if self.nb_fruits == 0: # If agent cleared the game
                return 110

            self.last_point_taken = 0
            return 10

        # See if the agent has been idle for too long
        if self.last_point_taken > 5:
            return -10

        return 0

    def check_ending(self):
        """
        Check if the game is over
        """
        if self.last_point_taken > 30:
            return True
        if self.nb_fruits == 0:
            return True
        for i, position in enumerate(self.positions[1:]):
            if self.positions[0] == position:
                return True
        for i, position in enumerate(self.old_positions[1:]):
            if self.positions[0] == position and self.old_positions[0] == self.positions[i+1]:
                return True
        return False

    def get_valid_moves(self, position):
        """
        Return the list of the possible moves starting from position.
        """
        valid_moves = []
        for move, action in self.action_map.items():
            try:
                new_position = self.check_position(index_sum(position, action))
                valid_moves.append(move)
            except InvalidIndex:
                pass
        return valid_moves

    def __hash__(self):
        """
        hash function for grid.
        The grid becomes unwriteable after calling this function
        """
        return sha1(self.grid.grid)

    def compute_distances(self):
        """
        Compute distance between every possible tiles (Djikstra's algorithm on all free tiles)
        """
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                start = (x, y)
                if self.grid[start] == 64:
                    continue
                flags = np.zeros(self.grid.shape)
                distances = np.zeros(self.grid.shape)
                queue = [start]
                flags[start] = 1
                while len(queue) > 0:
                    position = queue.pop()
                    moves = self.get_valid_moves(position)
                    new_positions = [index_sum(position, self.action_map[move]) for move in moves]
                    for new_position in new_positions:
                        new_position = self.check_position(new_position)
                        if not flags[new_position]:
                            flags[new_position] = 1
                            distances[new_position] = distances[position] + 1
                            queue = [new_position] + queue
                            self.distances[start] = distances
