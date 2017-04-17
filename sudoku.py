import numpy as np
from itertools import combinations
dim = 9
no_cells = dim*dim

class Board:
    def __init__(self,vals=None):
        self.board_matrix = np.zeros(shape=(dim,dim),dtype=np.uint8)
        if type(vals) == str:
            self.board_matrix = np.fromfile(vals)
        elif type(vals) == type(np.ndarray([])):
            self.board_matrix = vals
        elif vals is not None:
            for x,y,val in vals:
                self.board_matrix[x,y] = val

        self.free_count = sum(sum((self.board_matrix==0)))
        self.moves = np.zeros(shape=(self.free_count,2))

    def move(self,k,x,y):
        self.moves[k] = x,y

    def fill_square(self,x,y,val):
        self.board_matrix[x,y] = val
        self.free_count -= 1

    def free_square(self,x,y):
        self.board_matrix[x,y] = 0
        self.free_count+=1



class Sudoku:
    def __init__(self,vals=None):
        self.board = Board(vals)
        self.ncandidates = 0
        self.grid_dict = {0:(0,3),1:(0,3),2:(0,3),3:(3,6),4:(3,6),5:(3,6),6:(6,9),7:(6,9),8:(6,9)}
        self.k = 0
        self.counts = [1]
        self.indices = np.indices((9,9)).T.reshape(81,2)
        self.finished = False
        self.poss_dict = {}
       
    def __repr__(self):
        return str(self.board.board_matrix)

    def construct_candidates(self):
        x,y = self.next_square()
        if x < 0 and y <0:
            return None,x,y
        possible_values = self.possible_values(x,y)
        return possible_values,x,y


    def next_square(self):
        '''
        Returns the most constrained square i.e. the one which has the fewest possible choices
        '''
        #TODO: Change this to correctly find the indices which are zero
        self.zeros = np.argwhere(self.board.board_matrix==0)
        self.counts = np.array([len(self.possible_values(x[0],x[1])) for x in self.zeros])
        if min(self.counts) == 0:
            return -1,-1
        else:
            return self.zeros[(self.counts==min(self.counts))][0]


    def possible_values(self,x,y,poss_vals=None):
        '''
        Returns a boolean array indicating the possible values for the given coordinates
        '''
        x_g = self.grid_dict[x]
        y_g = self.grid_dict[y]

        x_vals = list(self.board.board_matrix[x,:])
        y_vals = list(self.board.board_matrix[:,y])
        grid_vals = list(self.board.board_matrix[x_g[0]:x_g[1],y_g[0]:y_g[1]].flatten())
        values = set(x_vals+y_vals+grid_vals)
        if poss_vals is None:
            vals = np.array([1,2,3,4,5,6,7,8,9])
        else:
            vals = poss_vals
        return [x for x in vals if x not in values] 
    def make_move(self,x,y,val):
        self.board.fill_square(x,y,val)

    def unmake_move(self,x,y):
        self.board.free_square(x,y)

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
            p_vals,x,y=self.construct_candidates()
            if p_vals is None:
                return
            self.k+=1
        for p in p_vals:
            self.make_move(x,y,p)
            a=self.backtrack()
            if self.finished:
                return a
            self.unmake_move(x,y)

class KillerSudoku(Sudoku):
    def __init__(self,sum_list=None):
        self.sum_map = np.zeros(shape=(dim,dim),dtype=np.uint8)
        self.sum_dict = {}
        self.running_total = np.zeros(shape=(dim,dim),dtype=np.uint8)
        for s in sum_list:
            for coords in s[1]:
                self.sum_dict[dim*coords[0]+coords[1]] = np.array([x for x in s[1]],dtype=np.uint8).T
                self.sum_map[coords[0],coords[1]] = s[0]
        super(KillerSudoku,self).__init__()
        self.val_array=self.construct_possible_array()
        
    
    def sum_values(self,total,count):
        return np.unique(np.array([x for x in combinations([1,2,3,4,5,6,7,8,9],count) if sum(x)==total],dtype=np.uint8))
    
    def construct_possible_array(self):
        val_array = np.zeros((dim,dim,dim),dtype=bool)
        for x in range(dim):
            for y in range(dim):
                total = self.sum_map[x,y]
                count = len(self.sum_dict[9*x+y].T)
                if self.board.board_matrix[x,y] !=0:
                    val_array[x,y,self.board.board_matrix[x,y]-1] = True
                else:
                    val_array[x,y,self.sum_values(total,count)-1] = True 
        return val_array
    
    def eliminate_loops(self,x,y,vals):
        x_poss = self.val_array[x,:]
        y_poss = self.val_array[:,y]
        
    #TODO: Make this smarter based on the possible values within the row and column
    def possible_values(self,x,y):
        total = self.sum_map[x,y]
        val_mask= self.val_array[x,y]
        vals = np.array([1,2,3,4,5,6,7,8,9])[val_mask]
        v = self.running_total[x,y]
        vals = vals[vals+v<=total]
        if len(vals) == 0:
            return []
        else:
            x_g = self.grid_dict[x]
            y_g = self.grid_dict[y]

            x_vals = self.val_array[x,:]
            y_vals = self.val_array[:,y]
            
            grid_vals = np.concatenate(self.val_array[x_g[0]:x_g[1],y_g[0]:y_g[1]]) 
            s = x_vals[np.sum(x_vals,axis=1)==1]
            t = y_vals[np.sum(y_vals,axis=1)==1]
            u = grid_vals[np.sum(grid_vals,axis=1)==1]
            
            val_bool = np.zeros(dim,dtype=bool)
            val_bool[vals-1] = True
            other_vals = np.bitwise_or.reduce(np.concatenate([~s,~t,~u,[val_bool]]),axis=0)
            return np.array([1,2,3,4,5,6,6,7,8])[other_vals] 
    def make_move(self,x,y,val):
        self.board.fill_square(x,y,val)
        xr = np.argwhere(self.val_array[:,y,val-1])
        yr = np.argwhere(self.val_array[x,:,val-1])
        self.val_array[x,:,val-1] = False
        self.val_array[:,y,val-1] = False
        self.val_array[x,y] = False
        self.val_array[x,y,val-1] = True
        ix,iy = self.sum_dict[9*x+y]
        self.running_total[ix,iy]+=val
        #return xr,yr
        
    def unmake_move(self,x,y):

        
        val = self.board.board_matrix[x,y]
        self.board.free_square(x,y)
        #self.val_array[reset_indices[0],y,val-1] = True
        #self.val_array[x,reset_indices[1],val-1] = True
        #self.val_array[x,y,val-1] = False
        ix,iy = self.sum_dict[9*x+y]
        self.running_total[ix,iy]-=val

    def next_square(self):
        zeros = np.argwhere(self.board.board_matrix==0)
        #counts = np.sum(self.val_array,axis=2)
        counts = []
        
        if False:
            return -1,-1
        else:
            '''
            args = counts[zeros[:,0],zeros[:,1]]
            for x,y in np.argwhere(counts==min(args)):
                if self.board.board_matrix[x,y] == 0:
                    return x,y
                    '''        
            for i,x in enumerate(zeros):
                c = len(self.possible_values(x[0],x[1]))
                if c == 0:
                    return -1,-1
                counts.append(c)
            counts = np.array(counts)
            return zeros[counts==min(counts)][0]
        
    
    def backtrack(self):
        if self.is_solved():
            self.finished = True
            return self.board.board_matrix
        else:
            p_vals,x,y=self.construct_candidates()
            if p_vals is None:
                return
            self.k+=1
        for p in p_vals:
            val_array = self.val_array.copy()
            self.make_move(x,y,p)
            a=self.backtrack()
            if self.finished:
                return a
            self.unmake_move(x,y)
            self.val_array = val_array.copy()
