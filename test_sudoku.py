import pytest

from sudoku import Sudoku
from utils import string_to_array
from numpy.testing import assert_array_equal
@pytest.fixture
def test_puzzle():
	return string_to_array('''003020600
900305001
001806400
008102900
700000008
006708200
002609500
800203009
005010300''')

@pytest.fixture
def test_solution():
	return string_to_array('''483921657
967345821
251876493
548132976
729564138
136798245
372689514
814253769
695417382''')


def test_solution_is_correct(test_puzzle, test_solution):
	su = Sudoku(test_puzzle)
	solution = su.backtrack()
	assert_array_equal(solution,test_solution)