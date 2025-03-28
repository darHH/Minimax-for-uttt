# Run the following cell to import utilities

import numpy as np
import time

from utils import State, Action, load_data

class StudentAgent:
    def __init__(self, depth_limit=4):
        self.depth_limit = depth_limit

    # evaluates a local
    def evaluate(self, state):

        evaluation = 0 
        # 0 for ongoing, 1 for p1, 2 for p2, 3 for draw
        local_board_status = state.local_board_status
        # if terminal, if win, return max + 1, else if lose return min - 1 and stop
        # if state.is_terminal():
        #     if state.terminal_utility() == 1:
        #         return 376 # max + 1
        #     elif state.terminal_utility() == 0:
        #         return -597 # max - 1
            
        # takes in local board which is 3x3 numpy
        def evaluate_local_board(local_board, state):

            evaluation = 0

            # 0 for empty, 1 for player 1 and 2 for player 2
            # for easier reference
            a1, a2, a3 = local_board[0]
            b1, b2, b3 = local_board[1]
            c1, c2, c3 = local_board[2]

            # 0 for ongoing, 1 for p1, 2 for p2, 3 for draw
            local_board_status = state.local_board_status

            # if global board sides are still available, then +EVAL for local sides to force opponent to play global sides, else -EVAL to prevent opponent from playing any board
            if a2 ==  1 and local_board_status[0][1] == 0:
                evaluation += 3
            elif a2 ==  1 and local_board_status[0][1] != 0:
                evaluation += -3
            if b1 == 1 and local_board_status[1][0] == 0:
                evaluation += 3
            elif b1 ==  1 and local_board_status[1][0] != 0:
                evaluation += -3
            if b3 == 1 and local_board_status[1][2] == 0:
                evaluation += 3
            elif b3 == 1 and local_board_status[1][2] != 0:
                evaluation += -3
            if c2 == 1 and local_board_status[2][1] == 0:
                evaluation += 3
            elif c2 == 1 and local_board_status[2][1] != 0:
                evaluation += -3
            
            # if global board corners are still available, then +0 

            # if global board center is still available, then -EVAL for center to not let opponent play center board 
            if b2 == 1 and local_board_status[1][1] == 0:
                evaluation += -6
            elif b2 == 1 and local_board_status[1][1] != 0:
                evaluation += -3
                
            return evaluation
        
        # for winning middle square, sides and corners, +EVAL respectively
        strategic_positions = [
            ((1, 1), 50),   # Center board (highest strategic value)
            ((0, 1), 30),   # Top side
            ((1, 0), 30),   # Left side
            ((1, 2), 30),   # Right side
            ((2, 1), 30),   # Bottom side
            ((0, 0), 20),   # Top-left corner
            ((0, 2), 20),   # Top-right corner
            ((2, 0), 20),   # Bottom-left corner
            ((2, 2), 20)    # Bottom-right corner
        ]

        for (row, col), value in strategic_positions:
            if local_board_status[row][col] == 1:
                evaluation += value
            elif local_board_status[row][col] == 2:
                evaluation -= value

        # and also evaluate each local board
        for i in range(0,3):
            for j in range(0,3): 
                evaluation += evaluate_local_board(state.board[i][j], state)

            # for getting two in a row, +40 or -40
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
            squares = [local_board_status[r, c] for (r, c) in line]
            
            # Count how many belong to each player (1 or 2) vs empty (0)
            count_1 = squares.count(1)  
            count_2 = squares.count(2) 
            count_0 = squares.count(0)  

            # If you have 2 in this line and 1 is empty => you can win in one move
            if count_1 == 2 and count_0 == 1 and count_2 == 0:
                evaluation += 40
            
            # If the opponent has 2 in this line and 1 is empty => they can win in one move
            if count_2 == 2 and count_0 == 1 and count_1 == 0:
                evaluation -= 40

        return evaluation
        
    
    def minimax_alpha_beta(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or state.is_terminal():
            return self.evaluate(state)

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