"""This module contains the player dynamics"""

from old_model.piece import *

class Player():
    def __init__(self, opp):
        self.__pawns = [] 
        for x in range(0,8):
            if opp == 1:
                self.__pawns.append(Pawn(opp, pos = [x, 6]))
                self.__king = King(opp, pos = [4, 7])
            elif opp == 2:
                self.__pawns.append(Pawn(opp, pos = [x, 1])) 
                self.__king = King(opp, pos = [4, 0])

            
    def pieces(self):
        pieces = []
        for pawn in self.__pawns: 
            pieces.append(pawn)
        pieces.append(self.__king)
        return pieces
        # need to add other pieces once done