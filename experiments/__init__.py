from os import listdir
from os.path import isfile, join, dirname

__dir = dirname(__file__)
types = tuple(f for f in listdir(__dir) if f.startswith('t_') and (isfile(join(__dir, f, '__init__.py'))))