
import numpy as np
import grid

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

	def __init__(self, id, initial_position, behaviour):
		self.id = id
		self.position = initial_position
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
		possible_moves = observation.get_valid_moves(self.position)
		random_draw = np.random.randint(0, len(possible_moves))
		return possible_moves[random_draw]

	def follow_move(self, observation):
		"""
		Return one of the possible moves which reduces the most the distance between the ghost and pacman.
		"""
		x_pacman, y_pacman = observation.positions[0]	# pacman's position
		x_ghost, y_ghost = self.position

		distance_after_move = observation.distances[x_ghost, y_ghost][x_pacman, y_pacman]

		for test_move in observation.get_valid_moves(self.position):
			x_test, y_test = self.position + observation.action_map[test_move]
			test_distance = observation.distances[x_test, y_test][x_pacman, y_pacman]
			if test_distance <= distance_after_move :
				move = test_move
				distance_after_move = test_distance
		return move

	def flee_move(self, observation):
		"""
		Return one of the possible moves which increases the most the distance between the ghost and the pacman.
		"""
		x_pacman, y_pacman = observation.positions[0]	# pacman's position
		x_ghost, y_ghost = self.position

		distance_after_move = observation.distances[x_ghost, y_ghost][x_pacman, y_pacman]

		for test_move in observation.get_valid_moves(self.position):
			x_test, y_test = self.position + observation.action_map[test_move]
			test_distance = observation.distances[x_test, y_test][x_pacman, y_pacman]
			if test_distance >= distance_after_move :
				move = test_move
				distance_after_move = test_distance
		return move