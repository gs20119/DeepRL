
from GridEnv.Gridworld import Gridworld # test this on python console
game = Gridworld(size=4, mode='static') # mode : 'static', 'player', 'random'
game.display()

game.makeMove('d')
game.makeMove('d')
game.makeMove('l')

game.display()
game.reward()
