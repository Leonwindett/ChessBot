# ChessBot: An AI-driven Chess Engine for Predicting Best Moves

## Description
ChessBot V1 leverages supervised machine learning, specifically using TensorFlow, to predict optimal moves during a game of chess. It follows a structured workflow to analyze game positions, process data, and make decisions based on learned strategies. All files related to the TensorFlow model can be found in the tensor_flow folder. Below is a breakdown of its functionality:

### 1. Data Input : Parsing Chess Games
ChessBot starts by reading chess game data in the form of PGN (Portable Game Notation) files, which contain detailed information about moves and outcomes. I have chosen chess games from various world tournaments played by the strongest grandmasters alive. These files are parsed into structured data using the python-chess library, which processes each move in a game sequence.

For each move, the current state of the chess board is saved as a tensor of shape (8, 8, 12), where the first two dimensions represent the 8x8 grid and the 12 channels represent different piece types (for both white and black). The subsequent move is saved as a tensor of shape (4096,) representing all possible moves (64 starting squares × 64 target squares). Both the board state and next move tensors are one-hot encoded to fit the input requirements of the neural network. This process is repeated for ~ 10<sup>5</sup> positions and stored as numpy arrays in the saved folder.

Special rules, such as castling, en passant, and pawn promotion, are handled by the python-chess library. Invalid moves and game-ending states like checkmate or stalemate are also flagged appropriately. The dataset is split with 80% for training and 20% for testing, with 10% of the training set used for validation to evaluate the model’s performance and prevent overfitting.

### 2. Model Architecture 

The neural network architecture consists of multiple layers, each designed to capture and process the complex patterns within the game. The model is trained to predict the next move based on the current board state via supervised learning from historical game data.

	•	2D Convolutional Layers: Used to scan the 8x8 board to detect strategic patterns like piece formations or tactical opportunities.
	•	Flatten Layer: Use to reduce the dimensions of the input to a 1D tensor that can be passed onto subsequent layers
	•	Dense (fully connected) Layers: ReLU activation functions are used to enable the network to model non-linear patterns in the data. The final layer uses a softmax activation function to produce a probability distribution over all possible moves. The model uses this output to predict the most likely next move (filtered by legality), given the current state of the board.

### 3. Training Process

For my optimizer I chose Adam - configured with a learning rate of 0.001. This was chosen over other optimizers because it offers a balance between speed and accuracy which suited this project best. With largely variant gradients in the learning process Adam's adjustable learning rates ensures this optimizer performs best for chess related tasks where some pieces may not move for multiple turns. 

The most suitable loss function to use was 'categorical cross-entropy' as this is a multi-class classification problem. 

The main metrics assessed were accuracy and validation loss. 

### 4. Problems Encountered

Throughout the learning process there has been many struggles as expected.

The main challenge has been storing enough training data to truly fine-tune the model. For context, ~ 450,000 positions and corresponding 'next moves' after vectorisation takes up 17GB. Moving my storage system for future machine learning projects to the cloud would hopefully fix this problem and allow me to grow my data base. 

Another problem faced was pushing the saved tensors to GitHub, again due to the file size. After adding the LFS attribute to git I am now able to update the saved folder however it takes an incredibly long time. If there is interest in the vectorised data please feel free to contact me and I can share the files an alternate way!

### 5. Learning Experience

Overall, this has been a great learning experience and has inspired me to delve deeper into the complex intricaces machine learning has to offer. I have planned out a couple Kaggle competitions to complete in the future and I hope to expand my knowledge at a rapid pace. 


