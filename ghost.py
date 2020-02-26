

class Ghost(object):
    
    behaviour_list = ['random', 'follow', 'flee', 'mixed']
	
	def __init__(self, id, initial_position, behaviour)
		self.id = id
		self.position = initial_position
		if (behaviour not in behaviour_list):
			print (f'No such behaviour. Must be one of {behaviour_list}.')
		else :
			self.behaviour = behaviour

	def get_id(self):
		return self.id
	
	def get_position(self):
		return self.position

	def get_behaviour(self):
		return self.behaviour

	def step(self, observation):
		current_position = observation.ghost.id
		if (self.behaviour == 'random'):
			mouv = self.random_mouv(observation)
		elif (self.behaviour == 'follow'):
			mouv = self.follow_mouv(observation)
		elif (self.behaviour == 'flee'):
			mouv = self.flee_mouv(observation)
		elif (self.behaviour == 'mixed'):
			random_draw = np.random()
			if (random_draw < p):
				mouv = self.random_mouv(observation)
			else :
				mouv = self.follow_mouv(observation)
		else :
			pass 
		return mouv
	
	# mouv : ('L','R','U','D')
	def random_mouv(self, observation):
		# TODO : returns a possible random mouv
		return

	def follow_mouv(self, observation):
		# TODO : returns a follow mouv
		return

	def flee_mouv(self, obersation):
		# TODO : returns a flee mouv
		return