# Run the following cell to import utilities

import numpy as np
import time

from utils import State, Action, load_data

class StudentAgent:
    def __init__(self, depth_limit=2):
        self.depth_limit = depth_limit
        """Instantiates your agent.
        """
        
    # takes in a local 3x3 board and returns 1, -1 for win and lost, 0 otherwise.
    def check_board_status(self, board: np.ndarray) -> int:
        for i in range(3):
            # Check rows
            if np.all(board[i, :] == 1):
                return 1 
            if np.all(board[i, :] == 2):
                return -1  
            
            # Check columns
            if np.all(board[:, i] == 1):
                return 1  
            if np.all(board[:, i] == 2):
                return -1 

        # Check diagonals
        if np.all(np.diagonal(board) == 1):
            return 1  
        if np.all(np.diagonal(board) == 2):
            return -1 

        if np.all(np.diagonal(np.fliplr(board)) == 1):
            return 1  
        if np.all(np.diagonal(np.fliplr(board)) == 2):
            return -1 
 
        return 0  # Draw

    # takes in a local board (3x3 matrix), and evaluates that local board.
    # can be taken by using board[0-2][0-2]
    # SINCE STATE IS IMMUTABLE, IMMUTABLE_LOCAL_BOARD WILL BE PASSED
    def evaluate_local_board(self, immutable_local_board):
        # flattened_board = [item for row in local_board for item in row]
        evaluation = 0

        # make it mutable
        local_board = immutable_local_board.copy()

        # assign values for easier access
        a1, a2, a3 = local_board[0] 
        b1, b2, b3 = local_board[1] 
        c1, c2, c3 = local_board[2]

        # first criteria: positional penalty and reward
        importance_of_square = [[0.2, 0.17, 0.2], [0.17, 0.22, 0.17], [0.2, 0.17, 0.2]]

        # check if win or lose first before local_board changes from values 1,0,2 to 1,0,-1
        evaluation += self.check_board_status(local_board) * 12

        for element in np.nditer(local_board):
            # change opponent position to be -1 for this and subsequent criterias
            if element == 2:
                element = -1
            evaluation += np.sum(local_board * importance_of_square)

        # second criteria: potential line formations for opponent
        if local_board[0].sum() == -2 or local_board[1].sum() == -2 or local_board[2].sum() == -2:
            evaluation -= 6
        if local_board[:, 0].sum() == -2 or local_board[:, 1].sum() == -2 or local_board[:, 2].sum() == -2:
            evaluation -= 6
        if a1 + b2 + c3 == -2 or c1 + b2 + a3 == -2:
            evaluation -= 6

        # third criteria: 2 square me 1 square opponent, potentially blocking me
        if (a1 + a2 == 2 and a3 == -1) or (a1 + a3 == 2 and a2 == -1) or (a2 + a3 == 2 and a1 == -1) or (b1 + b2 == 2 and b3 == -1) or (b1 + b3 == 2 and b2 == -1) or (b2 + b3 == 2 and b1 == -1) or (c1 + c2 == 2 and c3 == -1) or (c1 + c3 == 2 and c2 == -1) or (c2 + c3 == 2 and c1 == -1) or (a1 + b1 == 2 and c1 == -1) or (a1 + c1 == 2 and  b1 == -1) or (b1 + c1 == 2 and a1 == -1) or (a2 + b2 == 2 and c2 == -1) or (a2 + c2 == 2 and  b2 == -1) or (b2 + c2 == 2 and a2 == -1) or (a3 + b3 == 2 and c3 == -1) or (a3 + c3 == 2 and  b3 == -1) or (b3 + c3 == 2 and a3 == -1) or (a1 + b2 == 2 and c3 == -1) or (a1 + c3 == 2 and b2 == -1) or (b2 + c3 == 2 and a1 == -1) + (a3 + b2 == 2 and c1 == -1) or (a3 + c1 == 2 and b2 == -1) or (b3 + c1 == 2 and a3 == -1):
            evaluation -= 9

        # fourth criteria: potential line formations for me 
                # second criteria: potential line formations for opponent
        if local_board[0].sum() == 2 or local_board[1].sum() == 2 or local_board[2].sum() == 2:
            evaluation += 6
        if local_board[:, 0].sum() == 2 or local_board[:, 1].sum() == 2 or local_board[:, 2].sum() == 2:
            evaluation += 6
        if a1 + b2 + c3 == 2 or c1 + b2 + a3 == 2:
            evaluation += 6

        # fifth criteria: 2 square opponent 1 square me, potentially blocking opponent
        if (a1 + a2 == -2 and a3 == 1) or (a1 + a3 == -2 and a2 == 1) or (a2 + a3 == -2 and a1 == 1) or (b1 + b2 == -2 and b3 == 1) or (b1 + b3 == -2 and b2 == 1) or (b2 + b3 == -2 and b1 == 1) or (c1 + c2 == -2 and c3 == 1) or (c1 + c3 == -2 and c2 == 1) or (c2 + c3 == -2 and c1 == 1) or (a1 + b1 == -2 and c1 == 1) or (a1 + c1 == -2 and  b1 == 1) or (b1 + c1 == -2 and a1 == 1) or (a2 + b2 == -2 and c2 == 1) or (a2 + c2 == -2 and  b2 == 1) or (b2 + c2 == -2 and a2 == 1) or (a3 + b3 == -2 and c3 == 1) or (a3 + c3 == -2 and  b3 == 1) or (b3 + c3 == -2 and a3 == 1) or (a1 + b2 == -2 and c3 == 1) or (a1 + c3 == -2 and b2 == 1) or (b2 + c3 == -2 and a1 == 1) + (a3 + b2 == -2 and c1 == 1) or (a3 + c1 == -2 and b2 == 1) or (b3 + c1 == -2 and a3 == 1):
            evaluation += 9

        return evaluation

    def evaluate_global_board(self, state):
        evaluation =  0
        importance_of_local_board = [[1.4, 1, 1.4], [1, 1.75, 1], [1.4, 1, 1.4]]
        if state.prev_local_action is not None:
            gx, gy = state.prev_local_action
            current_local_board = state.board[gx, gy]
        else:
            current_local_board = (-1, -1)
        
        # iterate through all 9 local boards
        for i in range (0, 3):
            for j in range (0, 3):
                evaluation  += self.evaluate_local_board(state.board[i][j] * 1.5 * importance_of_local_board[i][j])
                # checks if it is current active board
                if (state.board[i][j] == current_local_board).all():
                    evaluation += self.evaluate_local_board(state.board[i][j] * importance_of_local_board[i][j])
                # checks if this board is won
                evaluation += self.check_board_status(state.board[i][j])

        # if global win 
        evaluation += self.check_board_status(state.local_board_status) * 5000

        # add bonus for overall state but must 
        evaluation += self.evaluate_local_board(state.local_board_status) * 150

        return evaluation
    
    def minimax_alpha_beta(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or state.is_terminal():
            return self.evaluate_global_board(state)

        valid_actions = state.get_all_valid_actions()

        if maximizing_player:
            max_eval = float('-inf')
            for action in valid_actions:
                next_state = state.change_state(action, in_place=False)
                eval = self.minimax_alpha_beta(next_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for action in valid_actions:
                next_state = state.change_state(action, in_place=False)
                eval = self.minimax_alpha_beta(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                
                if beta <= alpha:
                    break
            return min_eval

    def choose_action(self, state):
        best_score = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        
        valid_actions = state.get_all_valid_actions()
        
        for action in valid_actions:
            next_state = state.change_state(action, in_place=False)

            score = self.minimax_alpha_beta(next_state, depth=self.depth_limit-1, alpha=alpha, beta=beta, maximizing_player=False)
            
            if score > best_score:
                best_score = score
                best_action = action
            
            alpha = max(alpha, score)
        
        return best_action