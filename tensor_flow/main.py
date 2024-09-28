import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import chess.pgn as cp
import os
import shutil
from tqdm import tqdm
import keras
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout, BatchNormalization, Activation
from functions import *

directory = "/Users/leonwindett/VS_CODE/Projects/ChessBot/tensor_flow/pgn_files/processing"
processed_folder = "tensor_flow/pgn_files/processed_data"
filename_in = "tensor_flow/saved/chess_pos.npy"
filename_out = "tensor_flow/saved/next_move.npy"

   
def vectorisation(game):
    board = game.board()
    mainline_moves = list(game.mainline_moves())

    game_states = []
    state_targets = []
    for index, move in enumerate(mainline_moves[:-1]):
    
        board.push(move)

        board_state = convert_to_matrix(board)
        game_states.append(board_state)

        next_move = mainline_moves[index + 1]
        _from = next_move.from_square
        _to = next_move.to_square
        target_vector = output_vec(_from, _to)
        state_targets.append(target_vector)

    return game_states, state_targets


def data_process(dir): # this processes data in the Data folder and returns two tensors in desired form

    input_tensor = None
    target_tensor = None
    first_game = True

    total_files = count_files(dir)

    if os.listdir(dir) is None:
        return None, None

    else:
        with tqdm(total = total_files, desc = "File Progress") as outer_pbar:
            for name in os.listdir(dir):

                pgn = open(f"tensor_flow/pgn_files/processing/{name}")
                total_games = pgn_game_count(name)

                with tqdm(total = total_games, desc = f"{name} Game Progress", leave = False) as inner_pbar:
                    for _i in range(total_games):
                        game = cp.read_game(pgn)
                        inputs, targets = vectorisation(game)

                        if first_game is True:
                            input_tensor = inputs
                            target_tensor = targets
                            first_game = False

                        else:
                            try: 
                                input_tensor = np.concatenate((input_tensor, inputs), axis = 0)
                                target_tensor = np.concatenate((target_tensor, targets), axis = 0)
                            except ValueError:
                                print(f"  Error in {name}")
                                continue
                        inner_pbar.update(1)

                    shutil.move(f"tensor_flow/pgn_files/processing/{name}", os.path.join(processed_folder, name))
                    outer_pbar.update(1)

            #To save new tensors uncomment this and choose file name:
            # save_tensor(input_tensor, "tensor_flow/saved/chess_pos.npy")
            # save_tensor(target_tensor, "tensor_flow/saved/next_move.npy")

            return input_tensor, target_tensor


def update(filename1, filename2):

    input, output = data_process(directory) # think about turning into tf tensors if performance needed

    saved_input = np.load(filename1)
    saved_output = np.load(filename2)

    if input is None and output is None:
        updated_input = saved_input
        updated_output = saved_output
        pass
    
    else: 
        updated_input = np.concatenate((saved_input, input), axis = 0)
        updated_output = np.concatenate((saved_output, output), axis = 0)

        np.save(filename_in, updated_input)
        np.save(filename_out, updated_output)

    print(f"Input shape: {get_array_shape(updated_input)}")
    print(f"Target shape: {get_array_shape(updated_output)}")


def check_saved(filename1, filename2):

    saved_input = np.load(filename1)
    saved_output = np.load(filename2)
    
    print(f"Input shape: {get_array_shape(saved_input)}")
    print(f"Target shape: {get_array_shape(saved_output)}")


def ml_prep(input, target):
    """This is the final step before data is ready to be passed into model
       Data is split into 80% training and 20% testing
       Further splitting will be made on the training data to validate the model as it learns"""

    input_train = input[: int(0.8 * len(input))]
    input_test = input[int(0.8 * len(input)): ]

    target_train = target[: int(0.8 * len(target))]
    target_test = target[int(0.8 * len(target)) : ]

    return (input_train, target_train), (input_test, target_test)


def model_construction(epochs, batch_size, file_in, file_target):

    input_tensor, target_tensor = np.load(file_in), np.load(file_target)
    (input_train, target_train), (input_test, target_test) = ml_prep(input_tensor, target_tensor)

    keras.backend.clear_session()
    model = Sequential()

    model.add(Input(shape=(8, 8, 12)))

    # model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    
    # model.add(MaxPooling2D(pool_size = (2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4096, activation = 'softmax'))

    model.compile(optimizer = Adam(learning_rate=0.001), 
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    model.summary()


    history = model.fit(input_train, 
                        target_train, 
                        epochs = epochs, 
                        batch_size = batch_size,
                        validation_split = 0.1,
                        verbose = 1)
    

    return model, history


def training_loss_plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(dpi = 100)
    ax.plot(epochs, loss, 'bo', label = 'Training loss')
    ax.plot(epochs, val_loss, 'b', label = 'Validation loss')
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    return fig


def training_acc_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(dpi = 100)
    ax.plot(epochs, acc, 'bo', label = 'Training accuracy')
    ax.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

    return fig


def predict_next_move(board):
    board_matrix = convert_to_matrix(board)
    predictions = model.predict(board_matrix)
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(predictions)[::-1]
    for move_index in sorted_indices:
        move = output_to_move(move_index)
        if move in legal_moves_uci:
            return move
    return None

if __name__ == "__main__":
   
    # update(filename_in, filename_out)

    # model, history = model_construction(epochs = 50, batch_size = 64, file_in = filename_in, file_out = filename_out)
    # model.save("tensor_flow/saved/chess_model1.keras")
    
    


    plt.show()

