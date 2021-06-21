"""
solver.py - a module for interfacing with a Solver class
"""
from abc import abstractmethod

from sudoku import Board

class BaseSolver:

	def __init__(self, board: Board):
		self._board = board

	@abstractmethod
	def solve(self):
		pass


class BacktrackSolver(BaseSolver):
	"""A class to solve sudoku problems via a back-tracking algorithm"""

	def solve(self):
		pass