import numpy as np
import os
import sys
import pygame


class Gui(object):
	"""
	Class used to display the game using a GUI.

	Arguments:
	----------
	grid: the grid attribute of a Grid object.
	row_number: number of rows in the grid
	column_number: number of columns in the grid
	screen: the pygame screen

	Class variables:
	----------
	color_map: mapping between the name of a color and its RGB representation.
	color_by_id: mapping between the ids on the grid and the RGB representation of the associated color.
	SQUARESIZE, RADIUS: parameters used for the display

	"""	
	color_map = {
		'blue' : (0,0,255),
		'black' : (0,0,0),
		'grey' : (128,128,128),
		'yellow' : (255,255,0),
		'red' : (255,0,0),
		'cyan' : (0,255,255),
		'purple' : (238,130,238),
		'orange' : (255,165,0)
	}

	color_by_id = {
		64 : 'blue',
		32 : 'orange',
		16 : 'purple',
		8 : 'cyan',
		4 : 'red',
		2 : 'yellow',
		1 : 'yellow',
		0 : 'black'
	}

	SQUARESIZE = 30
	RADIUS = 5

	def __init__(self, grid):
		# pylint: disable=no-member
		pygame.init()
		os.environ['SDL_VIDEO_WINDOW_POS'] = "5,25"
		self.grid = grid.grid
		self.row_number, self.column_number = self.grid.shape
		# print (self.grid.shape)
		self.screen = pygame.display.set_mode((self.column_number*self.SQUARESIZE, (self.row_number+1)*self.SQUARESIZE))


		self.font = pygame.font.Font(None, 24)

		self.max_score = grid.nb_fruits
		self.score = 0

	def render(self):
		"""
		Update the display according to self.grid.
		"""
		for r in range(self.row_number):
			for c in range(self.column_number):
				value = np.binary_repr(self.grid[r,c], width=7)
				# print (value)
				if int(value[0]):
					self.draw_wall(r,c)
				else:
					self.draw_floor(r,c)
					if int(value[6]):
						self.draw_fruit(r,c)
					for index in range (1, len(value)-1):
						if int(value[6-index]):
							self.draw_monster(r,c,self.color_by_id[2**index])
							continue
							
		self.score = self.max_score - np.sum(self.grid & 1)
		self.text = self.font.render("Score : " + str(self.score) , True, (255, 0, 0), (0,0,0))
		textRect = self.text.get_rect()  
		textRect.center = (50, 20) 
		self.screen.blit(self.text, textRect)

		pygame.display.update()
		return

	def draw_floor(self, row_number, column_number):
		"""
		Draw a black tile at row_number, column_number.
		"""
		x, y = column_number*self.SQUARESIZE, (row_number+1)*self.SQUARESIZE
		pygame.draw.rect(self.screen, self.color_map['black'], (x, y, self.SQUARESIZE, self.SQUARESIZE))
		return

	def draw_wall(self, row_number, column_number):
		"""
		Draw a blue tile at row_number, column_number.
		"""
		x, y = column_number*self.SQUARESIZE, (row_number+1)*self.SQUARESIZE
		pygame.draw.rect(self.screen, self.color_map['blue'], (x, y, self.SQUARESIZE, self.SQUARESIZE))
		return

	def draw_fruit(self, row_number, column_number):
		"""
		Draw a small yellow fruit at row_number, column_number.
		"""
		x, y = int(column_number*self.SQUARESIZE + self.SQUARESIZE/2), int((row_number+1)*self.SQUARESIZE + self.SQUARESIZE/2)
		pygame.draw.circle(self.screen, self.color_map['yellow'], (x,y), self.RADIUS)
		return

	def draw_monster(self, row_number, column_number, color_name):
		"""
		Draw a monster (colored circle using the color associated to color_name) at row_number, column_number.
		"""
		x, y = int(column_number*self.SQUARESIZE + self.SQUARESIZE/2), int((row_number+1)*self.SQUARESIZE + self.SQUARESIZE/2)
		pygame.draw.circle(self.screen, self.color_map[color_name], (x,y), self.RADIUS*2)
		return


if __name__ == "__main__":
	from pacman import Env
	from ghost import Ghost

	env = Env(gui_display=True)
	env.render()

	action = 'L'

	while True :
		pygame.time.wait(250)
		
		legal = env.grid.get_valid_moves(env.grid.positions[0])

		events = pygame.event.get()
		for event in events:
			# pylint: disable=no-member
			if event.type == pygame.QUIT:
				sys.exit()

			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT and "L" in legal:
					action = "L"
				elif event.key == pygame.K_RIGHT and "R" in legal:
					action = "R"
				elif event.key == pygame.K_UP and "U" in legal:
					action = "U"
				elif event.key == pygame.K_DOWN and "D" in legal:
					action = "D"

		if action in legal :
			obs, reward, ended, info = env.step(action)
			env.gui.score += reward
			if ended :
				print(env.gui.score)
				break
		env.render()

