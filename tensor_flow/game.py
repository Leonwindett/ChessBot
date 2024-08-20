""" Module that contains all of the game mechanics"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from piece import * 
from player import Player
from functions import convert

class Game():

    
    def __init__(self, player1 = Player(1), player2 = Player(2)):
        self.__p1 = player1
        self.__p2 = player2

        self.__move_tot = 0

        #initialising chess board internals

        self.__board = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
            ]

        #initialising chess board visual 
        chessboard = np.zeros((8, 8))

        chessboard[1::2, ::2] = 1
        chessboard[::2, 1::2] = 1

        colors = [(0.8, 0.6, 0.4), (0.4, 0.2, 0.1)]  
        cmap = mcolors.ListedColormap(colors)
        x, y = np.meshgrid(np.arange(9), np.arange(9))

        plt.ion()
        
        self.__fig, self.__ax = plt.subplots(figsize=(6, 6))
        self.__ax.pcolormesh(x, y, chessboard, cmap=cmap, edgecolors='k', linewidth=0.1)
        for piece in self.__p1.pieces() + self.__p2.pieces():
            self.__ax.add_patch(piece.patch())
            self.__board[piece.pos()[1]][piece.pos()[0]] = piece
        self.__ax.invert_yaxis()
        self.__ax.axis('off')  
        # print(board)

    def start(self):
        print("""
              Welcome to PyChess by Leon.
              Please input your move in the form e4-e5 where e4 is your pieces starting square and e5 is the destination square
              If you have entered an invalid move you will be prompted to think again 
              May the best player win!
              """)
        while True:
            for row in self.__board:
                print(row)
            if self.__move_tot % 2 == 0:
                move = input("White to move: ")
                opp = 1
                if move == 'R':
                    print("Black is victorious via resignition!!")
                    break
            else:
                move = input("Black to move: ")
                opp = 2
                if move == 'R':
                    print("White is victorious via resignition!!")
                    break

            initial = convert(move[:2])
            final = convert(move[-2:])

            print(initial, final)
            i1, j1 = initial[1], initial[0]
            i2, j2 = final[1], final[0]

            piece1 = self.__board[i1][j1]
            piece2 = self.__board[i2][j2]

            print(piece1, piece2)
            

            if piece1 == 0: 
                print("Please move a piece! ")
                continue
            else: 
                if piece1.move(final) == False:
                    
                   print("Please enter a valid move lol! ")
                   continue
                else:
                    if piece2 == 0:
                       self.__ax.patches.remove(piece1.patch())
                       piece1.move(final)
                       self.__ax.add_patch(piece1.patch())
                       self.__fig.canvas.draw()
                       self.__board[i1][j1], self.__board[i2][j2] = self.__board[i2][j2], self.__board[i1][j1]
                       self.__move_tot += 1
                    else: 
                        if piece2.opp() == opp:
                            print("Please enter a valid move! ")
                            continue
                        else:
                            piece1.patch().remove()
                            piece1.take(final)
                            piece2.patch().remove()
                            self.__ax.add_patch(piece1.patch())
                            self.__fig.canvas.draw()
                            
                            self.__board[piece2.pos()[1]][piece2.pos()[0]] = 0
                            self.__board[i1][j1], self.__board[i2][j2] = self.__board[i2][j2], self.__board[i1][j1]
                            self.__move_tot += 1
                        

                         
        
        plt.ioff()
        plt.close(self.__fig)
                     
                

