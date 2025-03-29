# Run the following cell to import utilities

import numpy as np
import time

from utils import State, Action, load_data

class StudentAgent:
    def __init__(self, depth_limit=4):
        self.depth_limit = depth_limit

    # evaluates a local
    def evaluate(self, state, weights):
        # weights = [1,2,2,2,3,2,1,2.5,2,1,3.4]
        a,b,c,d,e,f,g,h,k,l,m = weights
        evaluation = 0 

        # if terminal, if win, return max + 1, else if lose return min - 1 and stop
        if state.is_terminal():
            if state.terminal_utility() == 1:
                return 125 # max + 1
            elif state.terminal_utility() == 0:
                return -130 # max - 1
            
        # takes in local board which is 3x3 numpy: 1 for p1, 2 for p2, 0 for empty
        def evaluate_local_board(local_board):

            evaluation = 0

            # FIRST LOCAL EVALUATION
            winning_lines = [
            # Rows
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            # Columns
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            # Diagonals
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
            ]

            for line in winning_lines:
                # Get the contents of these three squares
                squares = [local_board[r, c] for (r, c) in line]
                
                # Count how many belong to each player (1 or 2) vs empty (0)
                count_1 = squares.count(1)  
                count_2 = squares.count(2) 
                count_0 = squares.count(0)  

                # If i have 1 in this line and 2 is empty 
                if count_1 == 1 and count_0 == 2:
                    evaluation += a

                # If opponent have 1 in this line and 2 is empty
                elif count_2 == 1 and count_0 == 2:
                    evaluation -= a

                # If I have 2 in this line and 1 is empty => I can win in one move
                elif count_1 == 2 and count_0 == 1:
                    evaluation += b
                    
                
                # If the opponent has 2 in this line and 1 is empty => they can win in one move
                elif count_0 == 1 and count_2 == 2:
                    evaluation -= b
                    

                # If I have 2 in this line and 1 is blocked => bad for me 
                elif count_1 == 2 and count_2 == 1:
                    evaluation -= c
                    

                # If opponent has 2 in this line and 1 is b locked => good for me
                elif count_1 == 1 and count_2 == 2:
                    evaluation += c
                    
                # If i win 
                elif count_1 == 3:
                    evaluation += d
                
                # if i lose
                elif count_2 == 3:
                    evaluation -= d

            # SECOND LOCAL EVALUATION
            strategic_positions = [
                (1, 1, e),  # Center
                (0, 0, f), (0, 2, f), (2, 0, f), (2, 2, f),  # Corners
                (0, 1, g), (1, 0, g), (1, 2, g), (2, 1, g)   # Sides
            ]

            # Strategic position evaluation
            for row, col, value in strategic_positions:
                if local_board[row, col] == 1:
                    evaluation += value
                elif local_board[row, col] == 2:
                    evaluation -= value

                
            return evaluation
        
        # ------ BACK TO GLOBAL EVALUATION --------
        # 0 for ongoing, 1 for p1, 2 for p2, 3 for draw
        local_board_status = state.local_board_status

        # and evaluate each local board
        for i in range(0,3):
            for j in range(0,3): 
                # add different weights to each local board
                if i == 1 and j == 1:
                    evaluation += h * evaluate_local_board(state.board[i][j])
                # for corners
                elif i == 0 and j == 0:
                    evaluation += k * evaluate_local_board(state.board[i][j])
                elif i == 0 and j == 2:
                    evaluation += k * evaluate_local_board(state.board[i][j])
                elif i == 2 and j == 0:
                    evaluation += k * evaluate_local_board(state.board[i][j])
                elif i == 2 and j == 2:
                    evaluation += k * evaluate_local_board(state.board[i][j])
                # for sides
                else: 
                    evaluation += l * evaluate_local_board(state.board[i][j])

        # and apply same logic from evaluate local board to global board (add slightly more weight for global board)
        evaluation += m * evaluate_local_board(local_board_status)

        # if middle global board is still ongoing, then avoid local middle squares so opponent cannot win middle global board
        if local_board_status[1][1] == 0:
            evaluation -= 2 * np.sum(state.board[:, :, 1, 1] == 1)

        return evaluation
        
    
    def minimax_alpha_beta(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or state.is_terminal():
            return self.evaluate(state, [1,2,2,2,3,2,1,2.5,2,1,3.4])

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

        # First move always take middle
        if state.prev_local_action == None:
            return (1,1,1,1)
        
        valid_actions = state.get_all_valid_actions()
        
        for action in valid_actions:
            next_state = state.change_state(action, in_place=False)

            score = self.minimax_alpha_beta(next_state, depth=self.depth_limit-1, alpha=alpha, beta=beta, maximizing_player=False)
            
            if score > best_score:
                best_score = score
                best_action = action
            
            alpha = max(alpha, score)
        
        return best_action