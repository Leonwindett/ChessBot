""" Module that contains all of the game mechanics"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from piece import * 
from player import Player

class Game():

    
    def __init__(self, player1 = Player(1), player2 = Player(2)):
        self.__p1 = player1
        self.__p2 = player2

        #initialising chess board
        chessboard = np.ones((8, 8))

        chessboard[1::2, ::2] = 0  
        chessboard[::2, 1::2] = 0  

        colors = [(0.8, 0.6, 0.4), (0.4, 0.2, 0.1)]  
        cmap = mcolors.ListedColormap(colors)
        x, y = np.meshgrid(np.arange(9), np.arange(9))
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pcolormesh(x, y, chessboard, cmap=cmap, edgecolors='k', linewidth=0.1)
        for piece in self.__p1.pieces() + self.__p2.pieces():
            ax.add_patch(piece.patch())
        ax.axis('off')  
        plt.show()
