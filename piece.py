"""This module stores all the class information on the chess pieces"""

import numpy as np
from matplotlib.patches import Circle

class Pawn():
    def __init__(self, opp, pos = [0,1]):
        self.__pos = np.array(pos)
        self.__moves = 0
        self.__opp = opp # 1 is white, 2 is black 
        if opp == 1:
            self.__patch = Circle(pos, radius = 0.25, fc = 'white')
        elif opp == 2:
            self.__patch = Circle(pos, radius = 0.25, fc = 'black')

    def pos(self):
        return self.__pos
    
    def opp(self):
        return self.__opp
    
    def patch(self):
        return self.__patch

    def move(self, new_pos):
        new_pos = np.array(new_pos)
        if self.__opp == 1:
            difference = new_pos - self.__pos
        elif self.__opp == 2:
            difference = self.__pos - new_pos

        av_diff = [[0, 1], [0,2]]
        if self.__moves == 0 and 0 <= new_pos[0] <= 7 and 0 <= new_pos[1] <= 7:
            if any(np.array_equal(difference, array) for array in av_diff):
                self.__pos = np.array(new_pos)
                self.__moves += 1
        elif any(np.array_equal(difference, array) for array in av_diff[0]) and 0 <= new_pos[0] <= 7 and 0 <= new_pos[1] <= 7:
            self.__pos = np.array(new_pos)
            self.__moves += 1
        else: 
            pass

    def take(self, new_pos):
        new_pos = np.array(new_pos)
        if self.__opp == 1:
            difference = new_pos - self.__pos
        elif self.__opp == 2:
            difference = self.__pos - new_pos
        
        av_diff = [[1, 1], [-1, 1]]
        if any(np.array_equal(difference, array) for array in av_diff) and 0 <= new_pos[0] <= 7 and 0 <= new_pos[1] <= 7:
            self.__pos = np.array(new_pos)
            self.__moves += 1
        else: 
            pass


    

# class King():
#     def __init__(self, pos = [0,])