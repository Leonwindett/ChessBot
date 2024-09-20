# ChessBot: An AI-driven Chess Engine for Predicting Best Moves

## Description
ChessBot V1 leverages supervised machine learning, specifically using TensorFlow, to predict optimal moves during a game of chess. It follows a structured workflow to analyze game positions, process data, and make decisions based on learned strategies. All files related to the TensorFlow model can be found in the tensor_flow folder. Below is a breakdown of its functionality:

### 1. Data Input : Parsing Chess Games
ChessBot starts by reading chess game data in the form of PGN (Portable Game Notation) files, which contain detailed information about moves and outcomes. These files are parsed into structured data using the python-chess library, which processes each move in a game sequence.

For each move, the current state of the chess board is saved as a tensor of shape (8, 8, 12), where the first two dimensions represent the 8x8 grid and the 12 channels represent different piece types (for both white and black). The subsequent move is saved as a tensor of shape (4096,) representing all possible moves (64 starting squares × 64 target squares). Both the board state and next move tensors are one-hot encoded to fit the input requirements of the neural network. This process is repeated for ~ 10<sup>4</sup> games and stored as numpy arrays in the saved folder.

Special rules, such as castling, en passant, and pawn promotion, are handled by the python-chess library. Invalid moves and game-ending states like checkmate or stalemate are also flagged appropriately. The dataset is split with 80% for training and 20% for testing, with 10% of the training set used for validation to evaluate the model’s performance and prevent overfitting.

### 2. Model Architecture 

The neural network architecture consists of multiple layers, each designed to capture and process the complex patterns within the game. The model is trained to predict the next move based on the current board state via supervised learning from historical game data.

	•	2D Convolutional Layers: Convolution layers are used to scan the 8x8 board to detect strategic patterns like piece formations or tactical opportunities.
	•	Flatten layer: Use to reduce the dimensions of the input to a 1D tensor that can be passed onto subsequent layers
	•	Dense (fully connected) Layers: ReLU activation functions enable  higher The final layer produces a probability distribution over all possible legal moves. The model uses this output to predict the most likely next move, given the current state of the board.
