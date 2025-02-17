"""This module is for useful functions"""

import numpy as np
import chess
import os


def convert_to_int(board):
    l = [None] * 64
    for sq in chess.scan_reversed(board.occupied_co[chess.WHITE]):  # Check if white
            l[sq] = board.piece_type_at(sq) 
    for sq in chess.scan_reversed(board.occupied_co[chess.BLACK]):  # Check if black
            l[sq] = board.piece_type_at(sq) + 6
    return [0 if v is None else v for v in l]


def convert_to_matrix(board):
    board_matrix = []
    for i in range(0,8):
        board_int = convert_to_int(board)[i * 8 : (i+1) * 8]
        row = []
        for piece in board_int:
            square = []
            for j in range(0,12):
                if piece == 0 or piece != (j + 1):
                     square.append(0)
                elif piece == (j + 1):
                     square.append(1)
            row.append(square)
            
        board_matrix.append(row)    
    return board_matrix #8x8x12


def output_vec(_from, _to):
    move_index = _from * 64 + _to
    vector = np.zeros(4096)
    vector[move_index] = 1
    return vector


def output_to_move(output_vector):
    for move_index, move in enumerate(output_vector):
        if move == 1:
            _to = move_index % 64
            _from = int((move_index - _to) / 64)

            from_an = chess.square_name(_from)
            to_an = chess.square_name(_to)
            return f"{from_an} {to_an}" # doesn't matter too much how this is returned as we only need board.push(move)

        else: 
             continue


def pgn_game_count(file_name):

    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for enc in encodings:
        try:
            with open(f"tensor_flow/pgn_files/processing/{file_name}", 'r', encoding=enc) as f:
                content = f.read()
        except UnicodeDecodeError:
             continue
        
    # with open(f"Data/{file_name}") as file:
    #     content = file.read()
        game_count = content.count("[Event ")
        return game_count
     

def get_array_shape(array):
    
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
    return array.shape


def save_tensor(data, filename):
    np.save(filename, data)


def count_files(directory):
    all_entries = os.listdir(directory)
    total_files = sum(1 for entry in all_entries if os.path.isfile(os.path.join(directory, entry)))

    return total_files

def tensor_chunk(file_name):
    tensor = np.load(file_name)

    chunk_size = 349500 # max is 349525
    total_size = tensor.shape[0]
    for i in range(0, total_size, chunk_size):
       chunk = tensor[i: i + chunk_size]
       np.save(f'{file_name[:-4]}{i//chunk_size}.npy', chunk)

def combine_tensor_chunk(path_to_tensors, tensor_type): #tensor type is str either chess_pos or next_move
    files = os.listdir(path_to_tensors)

    matching_files = [f for f in files if tensor_type in f and f.endswith('.npy')]
    
    combined_tensor = None
    
    for file in matching_files:

        tensor = np.load(f"tensor_flow/saved/{file}")
        if combined_tensor is None: 
            combined_tensor = tensor
        else: 
            combined_tensor = np.concatenate((combined_tensor, tensor), axis = 0)

    return combined_tensor