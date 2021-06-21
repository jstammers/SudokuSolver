import numpy as np

dim = 9


class Board:
    def __init__(self, vals=None):
        self.board_matrix = np.zeros(shape=(dim, dim), dtype=np.uint8)
        if isinstance(vals, str):
            self.board_matrix = np.fromfile(vals)

        elif isinstance(vals, type(np.ndarray([]))):
            self.board_matrix = vals
        elif vals is not None:
            for x, y, val in vals:
                self.board_matrix[x, y] = val

        self.free_count = sum(sum((self.board_matrix == 0)))
        self.moves = np.zeros(shape=(self.free_count, 2))

    def move(self, k, x, y):
        self.moves[k] = x, y

    def fill_square(self, x, y, val):
        self.board_matrix[x, y] = val
        self.free_count -= 1

    def free_square(self, x, y):
        self.board_matrix[x, y] = 0
        self.free_count += 1