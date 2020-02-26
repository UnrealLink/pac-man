import numpy as np

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

    def __init__(self, grid_size=(25,25), player_spawn=(12, 12), ghost_spawn=(10, 12)):
        self.grid = np.ones(grid_size)
        self.grid[0 , :] = 64
        self.grid[-1, :] = 64
        self.grid[:,  0] = 64
        self.grid[:, -1] = 64
        self.player_spawn = player_spawn
        self.ghost_spawn  = ghost_spawn
        self.positions = [(0, 0)]*5   # 0: player, 1-4: ghosts

    def update(self, actions):
        """ 
        Update the grid according to agent and ghosts' actions.
        actions is a char list of size 5 containing the action ('U', 'D', 'R', 'L')
        """
        for i, action in enumerate(actions):
            self.grid[self.positions[i]] -= 2**(i+1)
            self.positions[i] += self.action_map[action]
            self.check_position(self.positions[i])
            self.grid[self.positions[i]] += 2**(i+1)


    def check_position(self, position):
        """
        Check if position is valid
        """
        x, y = position
        if x < 0 or x >= 25 or y > 0 or y >= 25:
            raise Exception("Invalid position")
        if self.grid[position] == 64:
            raise Exception("Invalid position")
        
