import numpy as np
import grid
import pygame
import sys
import math

pygame.init()




def draw_board(self):

	BLUE = (0,0,255)
	BLACK = (0,0,0)
	RED = (255,0,0)
	YELLOW = (255,255,0)
	CYAN = (0,255,255)
	GREY = (128,128,128)
	PURPLE = (238,130,238)
	ORANGE = (255,165,0)

	SQUARESIZE = 30
	RADIUS = 5

	board = self.grid

	ROW_COUNT, COLUMN_COUNT = board.shape

	width = COLUMN_COUNT * SQUARESIZE
	height = (ROW_COUNT+1) * SQUARESIZE

	size = (width, height)


	screen = pygame.display.set_mode(size)


	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):
			pygame.draw.rect(screen, BLACK, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
			if board[r,c] == 1 :
				pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
			elif board[r,c] == 2 :
				pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
			elif board[r,c] == 4 :
				pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
			elif board[r,c] == 8 :
				pygame.draw.circle(screen, CYAN, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
			elif board[r,c] == 16 :
				pygame.draw.circle(screen, ORANGE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
			elif board[r,c] == 32 :
				pygame.draw.circle(screen, PURPLE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
			elif board[r,c] == 64 :
				pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
			else : 
				case = np.binary_repr(board[r,c], width=7)
				if case[2] == '1' :
					pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
				elif case[3] == '1' :
					pygame.draw.circle(screen, CYAN, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
				elif case[4] == '1' :
					pygame.draw.circle(screen, ORANGE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)
				elif case[5] == '1' : 
					pygame.draw.circle(screen, PURPLE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS*2)



	
	pygame.display.update()



myfont = pygame.font.SysFont("monospace", 75)


draw_board(game.grid)
pygame.display.update()

while not game_over:

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()