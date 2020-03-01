import numpy as np
# import pygame
import sys
import math

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

    def __init__(self, board="board.txt", player_spawn=(15, 12), ghost_spawn=(9, 12)):
        with open(board, 'r') as board_file:
            self.grid = np.array([line.split() for line in board_file.readlines()], dtype=np.int8)
        self.player_spawn = player_spawn
        self.ghost_spawn  = ghost_spawn
        self.positions = [self.player_spawn] + [self.ghost_spawn]*4   # 0: player, 1-4: ghosts 
        for i, position in enumerate(self.positions):
            self.grid[position] = self.grid[position] + 2**(i+1)
        self.nb_fruits = 257
        self.distances = {}
        self.compute_distances()

        # SQUARESIZE = 30
        # ROW_COUNT, COLUMN_COUNT = self.grid.shape
        # screen = pygame.display.set_mode((COLUMN_COUNT * SQUARESIZE, ROW_COUNT * SQUARESIZE))
        # pygame.init()


    def update(self, actions):
        """ 
        Update the grid according to agent and ghosts' actions.
        actions is a char list of size 5 containing the action ('U', 'D', 'R', 'L')
        return (reward, ended) tuple.
        """
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
        if self.grid[self.positions[0]] & 1:
            self.grid[self.positions[0]] = self.grid[self.positions[0]] - 1
            self.nb_fruits -= 1
            if self.nb_fruits == 0:
                return 101
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
        for move, action in self.action_map.items():
            try:
                new_position = self.check_position(index_sum(position, action))
                valid_moves.append(move)
            except InvalidIndex:
                pass
        return valid_moves

    def compute_distances(self):
        """
        Compute distance between every possible tiles
        """
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                start = (x, y)
                if self.grid[start] == 64:
                    continue
                flags = np.zeros(self.grid.shape)
                distances = np.zeros(self.grid.shape)
                queue = [start]
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


    # def draw_board(self):

    #     BLUE = (0,0,255)
    #     BLACK = (0,0,0)
    #     RED = (255,0,0)
    #     YELLOW = (255,255,0)
    #     CYAN = (0,255,255)
    #     GREY = (128,128,128)
    #     PURPLE = (238,130,238)
    #     ORANGE = (255,165,0)

    #     SQUARESIZE = 30
    #     RADIUS = 5

    #     board = self.grid

    #     ROW_COUNT, COLUMN_COUNT = board.shape

    #     width = COLUMN_COUNT * SQUARESIZE
    #     height = ROW_COUNT * SQUARESIZE

    #     size = (width, height)


    #     screen = pygame.display.set_mode(size)


    #     for c in range(COLUMN_COUNT):
    #         for r in range(ROW_COUNT):
    #             pygame.draw.rect(screen, BLACK, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
    #             if board[r,c] == 1 :
    #                 pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    #             elif board[r,c] == 2 :
    #                 pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
    #             elif board[r,c] == 4 :
    #                 pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
    #             elif board[r,c] == 8 :
    #                 pygame.draw.circle(screen, CYAN, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
    #             elif board[r,c] == 16 :
    #                 pygame.draw.circle(screen, ORANGE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
    #             elif board[r,c] == 32 :
    #                 pygame.draw.circle(screen, PURPLE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
    #             elif board[r,c] == 64 :
    #                 pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
    #             else : 
    #                 case = np.binary_repr(board[r,c], width=7)
    #                 if case[2] == '1' :
    #                     pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
    #                 elif case[3] == '1' :
    #                     pygame.draw.circle(screen, CYAN, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
    #                 elif case[4] == '1' :
    #                     pygame.draw.circle(screen, ORANGE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
    #                 elif case[5] == '1' : 
    #                     pygame.draw.circle(screen, PURPLE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)

    #         pygame.display.update()
