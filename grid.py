import numpy as np
np.set_printoptions(linewidth=100)

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
        'U': (1, 0),
        'D': (-1, 0),
        'R': (0, 1),
        'L': (0, -1)
    }

    def __init__(self, player_spawn=(15, 12), ghost_spawn=(9, 12)):
        with open("board.txt", 'r') as board_file:
            self.grid = np.array([line.split() for line in board_file.readlines()], dtype=np.int8)
        self.player_spawn = player_spawn
        self.ghost_spawn  = ghost_spawn
        self.positions = [(0, 0)]*5   # 0: player, 1-4: ghosts
        self.positions[0] = self.player_spawn
        self.nb_fruits = 255 # TODO

    def update(self, actions):
        """ 
        Update the grid according to agent and ghosts' actions.
        actions is a char list of size 5 containing the action ('U', 'D', 'R', 'L')
        return (reward, ended) tuple.
        """
        for i, action in enumerate(actions):
            self.grid[self.positions[i]] = self.grid[self.positions[i]] - 2**(i+1)
            self.positions[i] = self.check_position(self.positions[i] + self.action_map[action])
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
            raise Exception("Invalid position")
        return position

    def compute_reward(self):
        """
        Check if the player got a reward for moving to his position
        Remove the fruit if needed
        """
        if self.grid[self.positions[0]] & 1:
            # reward
            self.grid[self.positions[0]] = self.grid[self.positions[0]] - 1
            self.nb_fruits -= 1
            return 1
        return 0

    def check_ending(self):
        """
        Check if the game is over
        """
        if self.nb_fruits == 0:
            return True
        for i, position in enumerate(self.positions[1:]):
            if self.positions[0] == position:
                return True
        return False

    def get_valid_moves(self, position):
        """
        Return the list of the possible moves starting from position.
        """
        valid_moves = []
        if self.grid[position[0] + 1, position[1]] != 64:
            valid_moves.append('U')
        if self.grid[position[0] - 1, position[1]] != 64:
            valid_moves.append('D')
        if self.grid[position[0], position[1] + 1] != 64:
            valid_moves.append('R')
        if self.grid[position[0], position[1] - 1] != 64:
            valid_moves.append('L')
        return valid_moves
