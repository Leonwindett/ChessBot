"""This module contains the player dynamics"""

from piece import Pawn

class Player():
    def __init__(self, opp):
        self.__pawns = [] 
        for x in range(0,8):
            if opp == 1:
                self.__pawns.append(Pawn(opp, pos = [x + 0.5, 1.5]))
            elif opp == 2:
                self.__pawns.append(Pawn(opp, pos = [x + 0.5, 6.5])) 

            
    def pieces(self):
        pieces = []
        for pawn in self.__pawns: 
            pieces.append(pawn)
        return pieces
        # need to add other pieces once done