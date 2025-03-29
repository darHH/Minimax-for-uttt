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

        # # if terminal, if win, return max + 1, else if lose return min - 1 and stop
        # if state.is_terminal():
        #     if state.terminal_utility() == 1:
        #         return 125 # max + 1
        #     elif state.terminal_utility() == 0:
        #         return -130 # max - 1
            
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
        
    def count_open_lines(self, local_board, player):
        opponent = 2 if player == 1 else 1
        lines = [
            [(0,0), (0,1), (0,2)],
            [(1,0), (1,1), (1,2)],
            [(2,0), (2,1), (2,2)],
            [(0,0), (1,0), (2,0)],
            [(0,1), (1,1), (2,1)],
            [(0,2), (1,2), (2,2)],
            [(0,0), (1,1), (2,2)],
            [(0,2), (1,1), (2,0)],
        ]
        count = 0
        for line in lines:
            values = [local_board[r, c] for r, c in line]
            if opponent not in values and (player in values):
                count += 1
        return count
    
    def choose_action(self, state):
        best_score = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')

        valid_actions = state.get_all_valid_actions()

        # First move always take middle
        if state.prev_local_action == None:
            return (1,1,1,1)

        # Return immediately
        for action in valid_actions:
            next_state = state.change_state(action, in_place=False)
            global_r, global_c, local_r, local_c = action

            # Win global immediately
            if next_state.is_terminal():
                if state.terminal_utility() == 1:
                    return action

            # Win local immediately
            if next_state.local_board_status[global_r][global_c] == 1:
                return action  

            # Block immediate threat
            if self.can_block_win(state.board[global_r][global_c], local_r, local_c):
                return action  
        
            # 3. Send opponent to a trap subgame
            num_pieces = np.sum(state.board != 0)
            if 10 < num_pieces < 60:  # mid-game
                # check for trap
                    target_local_status = state.local_board_status[local_r][local_c]

                    if target_local_status == 0:
                        local_board = state.board[local_r][local_c]

                        my_threats = self.count_open_lines(local_board, player=1)
                        opponent_threats = self.count_open_lines(local_board, player=2)
                    
                        if opponent_threats == 0 and my_threats >= 1:
                            return action 
                    
        # Strategic move prioritization
        prioritized_actions = self.prioritize_actions(state, valid_actions)

        for action in prioritized_actions:
            next_state = state.change_state(action, in_place=False)

            score = self.minimax_alpha_beta(next_state, depth=self.depth_limit-1, alpha=alpha, beta=beta, maximizing_player=False)
            
            if score > best_score:
                best_score = score
                best_action = action
            
            alpha = max(alpha, score)
        
        return best_action
    
    def prioritize_actions(self, state, valid_actions):
        action_scores = []
        
        local_status = state.local_board_status
        
        for action in valid_actions:
            global_r, global_c, local_r, local_c = action
            score = 0
            
            # # 1. Prioritize winning moves in any local board
            # next_state = state.change_state(action, in_place=False)
            # if next_state.local_board_status[global_r, global_c] == 1:
            #     score += 1000
                
            # # 2. Prioritize blocking opponent's winning moves
            # if self.can_block_win(state.board[global_r][global_c], local_r, local_c):
            #     score += 800
                
            # 3. Prefer center of local boards (especially early game)
            if global_r == 1 and global_c == 1:
                score += 100
            
            # 4a. Avoid sending opponent to finished boards (gives them choice)
            next_local_board = (local_r, local_c)
            if local_status[next_local_board] != 0:  # Board is finished
                score -= 200
                
            # 4b. Prefer corner subgames in early game
            num_pieces = np.sum(state.board != 0)
            if num_pieces < 16:
                if (global_r == 0 and global_c == 0) or \
                (global_r == 0 and global_c == 2) or \
                (global_r == 2 and global_c == 0) or \
                (global_r == 2 and global_c == 2):
                    score += 150
                    
            # 4c. Consider side subgames in mid-game
            elif num_pieces < 40:
                if (global_r == 0 and global_c == 1) or \
                (global_r == 1 and global_c == 0) or \
                (global_r == 1 and global_c == 2) or \
                (global_r == 2 and global_c == 1):
                    score += 50
                    
            action_scores.append((action, score))
        
        return [a for a, s in sorted(action_scores, key=lambda x: x[1], reverse=True)]

    def can_block_win(self, local_board, row, col):
        temp_board = local_board.copy()
        temp_board[row, col] = 1
        
        # Check if opponent would have won if we didn't block
        temp_board_without_block = local_board.copy()
        temp_board_without_block[row, col] = 2
        
        # Check if this local position would have completed a line for opponent
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
            if (row, col) in line:
                # Check if this line would have been a win for opponent
                squares = [temp_board_without_block[r, c] for r, c in line]
                if squares.count(2) == 3:
                    return True
                    
        return False