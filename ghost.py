
import numpy as np
from utils import index_sum

class Ghost(object):
	"""
	Class used to represent the different ghosts.

	Arguments:
	----------
	id : the id of the ghost 
	position : the position of the ghost on the grid 
	behaviour : the way the ghost behaves

	Class variables:
	----------
	behaviour list: the list of the different possible behaviours
		- "random": the ghost moves randomly
		- "follow": the ghost tries to get closer to the pacman
		- "flee": the ghost tries to get as far as possible from the pacman
		- "mixed": at each step, the ghost behaves randomly with probability 0.5 
					and tries to get closer to the pacman with probability 0.5
	
	"""	
	behaviour_list = ['random', 'follow', 'flee', 'mixed']

	def __init__(self, id, behaviour):
		self.id = id
		if (behaviour not in self.behaviour_list):
			raise Exception(f"No such behaviour. Must be one of {self.behaviour_list}.")
		else :
			self.behaviour = behaviour

	def step(self, observation):
		"""
		Return the move the ghost will perform depending on its position and pacman's position.
		Is one of 'L', 'R', 'U', 'D'.
		"""
		if (self.behaviour == 'random'):
			move = self.random_move(observation)
		elif (self.behaviour == 'follow'):
			move = self.follow_move(observation)
		elif (self.behaviour == 'flee'):
			move = self.flee_move(observation)
		elif (self.behaviour == 'mixed'):
			random_draw = np.random.random()
			if (random_draw < 0.5):
				move = self.random_move(observation)
			else :
				move = self.follow_move(observation)
		else :
			raise (f"No such behaviour. Must be one of {self.behaviour_list}.")
		return move

	def random_move(self, observation):
		"""
		Return a random possible move.
		"""
		possible_moves = observation.get_valid_moves(observation.positions[self.id])
		random_draw = np.random.randint(0, len(possible_moves))
		return possible_moves[random_draw]

	def follow_move(self, observation):
		"""
		Return one of the possible moves which reduces the most the distance between the ghost and pacman.
		"""
		x_pacman, y_pacman = observation.positions[0]	# pacman's position

		move = self.random_move(observation)
		x_new, y_new = observation.check_position(index_sum(observation.positions[self.id], observation.action_map[move]))
		distance_after_move = observation.distances[x_new, y_new][x_pacman, y_pacman]

		for test_move in observation.get_valid_moves(observation.positions[self.id]):
			x_test, y_test = observation.check_position(index_sum(observation.positions[self.id], observation.action_map[test_move]))
			test_distance = observation.distances[x_test, y_test][x_pacman, y_pacman]
			if test_distance < distance_after_move :
				move = test_move
				distance_after_move = test_distance
		return move

	def flee_move(self, observation):
		"""
		Return one of the possible moves which increases the most the distance between the ghost and the pacman.
		"""
		x_pacman, y_pacman = observation.positions[0]	# pacman's position

		move = self.random_move(observation)
		x_new, y_new = observation.check_position(index_sum(observation.positions[self.id], observation.action_map[move]))
		distance_after_move = observation.distances[x_new, y_new][x_pacman, y_pacman]

		for test_move in observation.get_valid_moves(observation.positions[self.id]):
			x_test, y_test = observation.check_position(index_sum(observation.positions[self.id], observation.action_map[test_move]))
			test_distance = observation.distances[x_test, y_test][x_pacman, y_pacman]
			if test_distance > distance_after_move :
				move = test_move
				distance_after_move = test_distance
		return move

if __name__ == "__main__":
	from grid import Grid

	grid = Grid()
	ghost1 = Ghost(1, 'random')
	ghost2 = Ghost(2, 'follow')
	ghost3 = Ghost(3, 'flee')
	ghost4 = Ghost(4, 'mixed')
	ghosts = [ghost1, ghost2, ghost3, ghost4]
	for i in range(10):
		actions = ['L' if i%2 else 'R'] + [ghost.step(grid) for ghost in ghosts]
		print(actions)
		grid.update(actions)
		print(grid.grid)
