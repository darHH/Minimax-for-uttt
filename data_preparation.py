import numpy as np
import torch

def augment_board(board):
    """
    Generate augmented boards using rotations and horizontal flips.
    
    Parameters:
        board (np.ndarray): A 2D numpy array representing the board (shape: [9,9]).
    
    Returns:
        List[np.ndarray]: A list of unique augmented board states.
    """
    augmented_boards = []
    
    # Apply rotations: 0, 90, 180, 270 degrees.
    for k in range(4):
        rotated = np.rot90(board, k)
        augmented_boards.append(rotated)
        # Also apply horizontal flip on the rotated board.
        flipped = np.fliplr(rotated)
        augmented_boards.append(flipped)
    
    # Remove duplicates if any (this is useful when the board is symmetric).
    unique_boards = []
    seen = set()
    for b in augmented_boards:
        # Convert the board to a tuple to make it hashable.
        b_tuple = tuple(b.flatten())
        if b_tuple not in seen:
            seen.add(b_tuple)
            unique_boards.append(b)
    return unique_boards

def prepare_data(states_4d, evaluations, augment=True):
    """
    Prepares the data by one-hot encoding the board states and optionally applying data augmentation.
    
    Parameters:
        states_4d (np.ndarray): Original board states with an attribute 'board'.
        evaluations (np.ndarray): The corresponding evaluations.
        augment (bool): If True, apply data augmentation using board symmetries.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (X_final, y_final)
    """
    augmented_boards_list = []
    augmented_evals_list = []
    
    for idx, state in enumerate(states_4d):
        # Assume each state has a .board attribute representing the board array.
        board_array = state.board.reshape(9, 9)  # Reshape if necessary
        
        # If augmentation is enabled, generate multiple versions of the board.
        if augment:
            boards = augment_board(board_array)
        else:
            boards = [board_array]
        
        for b in boards:
            # One-hot encode: each board is reshaped into 81 tiles with 3 channels.
            one_hot = np.zeros((81, 3))
            for tile_idx, tile_val in enumerate(b.flatten()):
                one_hot[tile_idx, tile_val] = 1
            # Flatten the one-hot encoded board into a vector of length 243.
            augmented_boards_list.append(one_hot.reshape(243))
            augmented_evals_list.append(evaluations[idx])
    
    X_final = np.array(augmented_boards_list)
    y_final = np.array(augmented_evals_list).reshape(-1, 1)
    
    return torch.tensor(X_final, dtype=torch.float32), torch.tensor(y_final, dtype=torch.float32)

# Usage example:
# X, y = prepare_data(states_array, evaluations_array, augment=True)