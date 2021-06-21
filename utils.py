import numpy as np


def string_to_array(sudoku_string: str) -> np.array:
	arr = []
	for line in sudoku_string.split('\n'):
		row = [int(x) for x in line]
		arr.append(row)
	return np.array(arr)
