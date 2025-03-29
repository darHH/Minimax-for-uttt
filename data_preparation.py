import numpy as np
import torch

def prepare_data(states_4d, evaluations):
    N = states_4d.shape[0]
    X_encoded = np.zeros((N, 81, 3))
    
    for idx, state in enumerate(states_4d):
        board_array = state.board.reshape(9, 9)  # Use `.board` attribute here!
        for tile_idx, tile_val in enumerate(board_array.flatten()):
            X_encoded[idx, tile_idx, tile_val] = 1  # One-hot encoding: 0,1,2

    X_final = X_encoded.reshape(N, 243)  # 81 tiles × 3 values per tile
    y_final = evaluations.reshape(N, 1)
    
    return torch.tensor(X_final, dtype=torch.float32), torch.tensor(y_final, dtype=torch.float32)