'''
sudoku.py : A sudoku program
'''
from itertools import combinations
import numpy as np
dim = 9
no_cells = dim*dim


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


class Sudoku:
    '''
    Represents a sudoku board
    '''

    def __init__(self, vals=None):
        self.board = Board(vals)
        self.ncandidates = 0
        self.grid_dict = {
            0: (0, 3),
            1: (0, 3),
            2: (0, 3),
            3: (3, 6),
            4: (3, 6),
            5: (3, 6),
            6: (6, 9),
            7: (6, 9),
            8: (6, 9)}
        self.k = 0
        self.counts = [1]
        self.indices = np.indices((9, 9)).T.reshape(81, 2)
        self.finished = False
        self.poss_dict = {}

    def __repr__(self):
        return str(self.board.board_matrix)

    def construct_candidates(self):
        x, y = self.next_square()
        if x < 0 and y < 0:
            return None, x, y
        possible_values = self.possible_values(x, y)
        return possible_values, x, y

    def next_square(self):
        '''
        Returns the most constrained square i.e. the one which has the fewest possible choices
        '''
        # TODO: Change this to correctly find the indices which are zero
        self.zeros = np.argwhere(self.board.board_matrix == 0)
        self.counts = np.array(
            [len(self.possible_values(x[0], x[1])) for x in self.zeros])
        if min(self.counts) == 0:
            return -1, -1
        else:
            return self.zeros[(self.counts == min(self.counts))][0]

    def possible_values(self, x, y, poss_vals=None):
        '''
        Returns a boolean array indicating the possible values for the given coordinates
        '''
        x_g = self.grid_dict[x]
        y_g = self.grid_dict[y]

        x_vals = list(self.board.board_matrix[x, :])
        y_vals = list(self.board.board_matrix[:, y])
        grid_vals = list(self.board.board_matrix
                         [x_g[0]: x_g[1],
                          y_g[0]: y_g[1]].flatten())
        values = set(x_vals+y_vals+grid_vals)
        if poss_vals is None:
            vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        else:
            vals = poss_vals
        return [x for x in vals if x not in values]

    def make_move(self, x, y, val):
        self.board.fill_square(x, y, val)

    def unmake_move(self, x, y):
        self.board.free_square(x, y)

    def is_solved(self):
        if self.board.free_count == 0:
            return True
        else:
            return False

    def backtrack(self):
        if self.is_solved():
            self.finished = True
            return self.board.board_matrix
        else:
            p_vals, x, y = self.construct_candidates()
            if p_vals is None:
                return
            self.k += 1
        for p in p_vals:
            self.make_move(x, y, p)
            a = self.backtrack()
            if self.finished:
                return a
            self.unmake_move(x, y)


class KillerSudoku(Sudoku):
    def __init__(self, sum_list=None):
        self.sum_map = np.zeros(shape=(dim, dim), dtype=np.uint8)
        self.sum_dict = {}
        self.running_total = np.zeros(shape=(dim, dim), dtype=np.uint8)
        for s in sum_list:
            for coords in s[1]:
                self.sum_dict[dim*coords[0]+coords[1]
                              ] = np.array([x for x in s[1]], dtype=np.uint8).T
                self.sum_map[coords[0], coords[1]] = s[0]
        super(KillerSudoku, self).__init__()
        self.val_array = self.construct_possible_array()

    def sum_values(self, total, count):
        return np.unique(np.array(
            [x
             for x in combinations(
                 [1, 2, 3, 4, 5, 6, 7, 8, 9],
                 count) if sum(x) == total],
            dtype=np.uint8))

    def construct_possible_array(self):
        val_array = np.zeros((dim, dim, dim), dtype=bool)
        for x in range(dim):
            for y in range(dim):
                total = self.sum_map[x, y]
                count = len(self.sum_dict[9*x+y].T)
                if self.board.board_matrix[x, y] != 0:
                    val_array[x, y, self.board.board_matrix[x, y]-1] = True
                else:
                    val_array[x, y, self.sum_values(total, count)-1] = True
        return val_array

    def eliminate_loops(self, x, y, vals):
        x_poss = self.val_array[x, :]
        y_poss = self.val_array[:, y]

    # TODO: Make this smarter based on the possible values within the row and
    # column
    def possible_values(self, x, y):
        total = self.sum_map[x, y]
        val_mask = self.val_array[x, y]
        vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[val_mask]
        v = self.running_total[x, y]
        vals = vals[vals+v <= total]
        if len(vals) == 0:
            return []
        else:
            x_g = self.grid_dict[x]
            y_g = self.grid_dict[y]

            x_vals = self.val_array[x, :]
            y_vals = self.val_array[:, y]

            grid_vals = np.concatenate(
                self.val_array[x_g[0]:x_g[1], y_g[0]:y_g[1]])

            x_vals[y] = np.zeros(dim, dtype=bool)
            y_vals[x] = np.zeros(dim, dtype=bool)
            grid_vals[(dim*x+y) % dim] = np.zeros(dim, dtype=bool)
            s = x_vals[np.sum(x_vals, axis=1) == 1]
            t = y_vals[np.sum(y_vals, axis=1) == 1]
            u = grid_vals[np.sum(grid_vals, axis=1) == 1]

            val_bool = np.ones(dim, dtype=bool)
            val_bool[vals-1] = False
            non_vals = [_ for _ in [s, t, u, val_bool] if _ is not None]

            other_vals = np.bitwise_or.reduce(
                np.concatenate([s, t, u, [val_bool]]), axis=0)
            return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[~other_vals]

    def make_move(self, x, y, val):
        self.board.fill_square(x, y, val)
        xr = np.argwhere(self.val_array[:, y, val-1])
        yr = np.argwhere(self.val_array[x, :, val-1])
        self.val_array[x, :, val-1] = False
        self.val_array[:, y, val-1] = False
        self.val_array[x, y] = False
        self.val_array[x, y, val-1] = True
        ix, iy = self.sum_dict[9*x+y]
        self.running_total[ix, iy] += val

    def unmake_move(self, x, y):

        val = self.board.board_matrix[x, y]
        self.board.free_square(x, y)
        ix, iy = self.sum_dict[9*x+y]
        self.running_total[ix, iy] -= val

    def next_square(self):
        zeros = np.argwhere(self.board.board_matrix == 0)
        counts = []

        if False:
            return -1, -1
        else:
            for i, x in enumerate(zeros):
                c = len(self.possible_values(x[0], x[1]))
                if c == 0:
                    return -1, -1
                counts.append(c)
            counts = np.array(counts)
            return zeros[counts == min(counts)][0]

    # def backtrack(self):
    #     if self.is_solved():
    #         self.finished = True
    #         return self.board.board_matrix
    #     else:
    #         p_vals, x, y = self.construct_candidates()
    #         if p_vals is None:
    #             return
    #         self.k += 1
    #     for p in p_vals:
    #         val_array = self.val_array.copy()
    #         self.make_move(x, y, p)
    #         a = self.backtrack()
    #         if self.finished:
    #             return a
    #         self.unmake_move(x, y)
    #         self.val_array = val_array.copy()


if __name__ == '__main__':
    arr = np.array([
[0,0,3,0,2,0,6,0,0],
[9,0,0,3,0,5,0,0,1],
[0,0,1,8,0,6,4,0,0],
[0,0,8,1,0,2,9,0,0],
[7,0,0,0,0,0,0,0,8],
[0,0,6,7,0,8,2,0,0],
[0,0,2,6,0,9,5,0,0],
[8,0,0,2,0,3,0,0,9],
[0,0,5,0,1,0,3,0,0]])

    k_list2 = [(16,[(0,0),(0,1),(1,0)]),
               (3,[(0,2),(0,3)]),
               (16,[(0,4),(0,5)]),
               (14,[(0,6),(0,7),(1,6),(1,5)]),
               (27,[(0,8),(1,8),(1,7),(2,7)]),
               (13,[(1,1),(1,2)]),
               (18,[(1,3),(2,3),(2,2),(2,1)]),
               (14,[(1,4),(2,4)]),
               (22,[(2,5),(2,6),(3,3),(3,4),(3,5)]),
               (17,[(2,8),(3,8),(4,8),(5,8),(6,8)]),
               (24,[(2,0),(3,0),(4,0),(5,0),(6,0)]),
               (23,[(3,1),(3,2),(4,1),(5,1),(5,2)]),
               (23,[(4,2),(4,3),(4,4),(4,5),(4,6)]),
               (34,[(3,6),(3,7),(4,7),(5,7),(5,6)]),
               (26,[(5,3),(5,4),(5,5),(6,3),(6,2)]),
               (17,[(6,1),(7,1),(7,0),(8,0)]),
               (6,[(6,4),(7,4)]),
               (22,[(6,7),(6,5),(6,6),(7,5)]),
               (8,[(7,6),(7,7)]),
               (24,[(7,2),(7,3),(8,2),(8,1)]),
               (6,[(8,3),(8,4)]),
               (17,[(8,5),(8,6)]),
               (15,[(7,8),(8,8),(8,7)])]

    k_list=[(13,[(0,0),(0,1),(0,2)]),
        (5,[(0,3)]),
        (11,[(0,4),(1,4)]),
        (1,[(0,5)]),
        (17,[(0,6),(0,7),(0,8)]),
        (23,[(1,0),(1,1),(2,0)]),
        (9,[(1,2),(2,1),(2,2)]),
        (7,[(1,3),(2,3)]),
        (35,[(1,5),(2,4),(2,5),(3,4),(3,5)]),
        (16,[(1,6),(2,6)]),
        (9,[(1,7),(2,7),(3,7)]),
        (9,[(1,8),(2,8)]),
        (10,[(3,0),(4,0)]),
        (20,[(3,1),(4,1),(5,1),(5,0)]),
        (12,[(3,2),(3,3),(4,3)]),
        (35,[(4,2),(5,2),(5,3),(6,2),(6,3)]),
        (13,[(3,6),(4,6)]),
        (3,[(3,8)]),
        (13,[(4,4),(5,4),(5,5)]),
        (2,[(4,5)]),
        (10,[(4,7),(4,8)]),
        (6,[(5,6),(5,7)]),
        (15,[(5,8),(6,8)]),
        (4,[(6,0)]),
        (11,[(6,1),(7,1),(7,0)]),
        (17,[(7,2),(7,3),(7,4)]),
        (13,[(6,4),(6,5),(7,5),(8,5)]),
        (9,[(6,6),(6,7)]),
        (15,[(8,0),(8,1),(8,2)]),
        (14,[(8,3),(8,4)]),
        (12,[(7,6),(8,6),(8,7)]),
        (14,[(7,7),(7,8)]),
        (2,[(8,8)])
        ]
    k2 = KillerSudoku(k_list)

    sol = k2.backtrack()

    print(sol)