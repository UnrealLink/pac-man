# pac-man

This is an implementation of Pac-Man with a slightly modified set of rules.
We designed two agents in order to clear the game, one using Approximate Q-Learning, the other using Deep Q-Network.

## Files Description

The *pacman.py* file implements the Gym AI Env object representing our Pac-Man game. The board itself is represented by a Grid object implemented in the *grid.py* file reading from either one of the *board.txt* and *board2.txt* files, while the ghost mecanisms are implemented in *ghosts.py*. A user interface is also implemented in *gui.py*.

The RL Agents we created can be found in the *DQN.py*, *Q_learning.py* and *Q_learning2.py* file. All three files can be launch to train models or evaluate ones, according to the options you specify in their main functions.

Some useful functions are implemented in *utils.py*

The *models* and *scores* folders contains *.txt* and *.pth* saved from training DQN agents. You should not modify them directly.

## How to use

If you want to train a model using Approximate Q-Learning, uncomment the corresponding lines in the main function at the end of *Q_learning.py* and run the file.

If you want to train a model using Deep Q-Networks, uncomment the corresponding lines in the main function at the end of *DQN.py* and run the file.

If you want to test your own skills against our Pac-Man game, just run the *gui.py* file. You will be able to control Pac-Man using your keyboard's arrows.