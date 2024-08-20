"""This module stores all the class information on the chess pieces"""

import numpy as np
from matplotlib.patches import Circle, Rectangle

class Pawn():
    def __init__(self, opp, pos):
        self.__pos = np.array(pos)
        self.__moves = 0
        self.__opp = opp # 1 is white, 2 is black 
        if opp == 1:
            self.__patch = Circle((self.__pos[0]+0.5, self.__pos[1]+0.5), radius = 0.25, fc = 'white')
        elif opp == 2:
            self.__patch = Circle((self.__pos[0]+0.5, self.__pos[1]+0.5), radius = 0.25, fc = 'black')

    def pos(self):
        return self.__pos
    
    def opp(self):
        return self.__opp
    
    def patch(self):
        return self.__patch

    def move(self, new_pos):
        new_pos = np.array(new_pos)
        if self.__opp == 1:
            difference = [self.__pos[0]-new_pos[0], self.__pos[1]-new_pos[1]]
        elif self.__opp == 2:
            difference = [new_pos[0]-self.__pos[0], new_pos[1]-self.__pos[1]]

        print(difference)

        av_diff = [[0, 1], [0, 2]]
        print(self.__moves)

        if self.__moves == 0:
            if np.array_equal([0, 1], difference) or np.array_equal([0, 2], difference):
                self.__pos = np.array(new_pos)
                if self.__opp == 1:
                    self.__patch = Circle((self.__pos[0]+0.5, self.__pos[1]+0.5), radius = 0.25, fc = 'white')
                if self.__opp == 2:
                    self.__patch = Circle((self.__pos[0]+0.5, self.__pos[1]+0.5), radius = 0.25, fc = 'black')
                self.__moves += 1
    
        elif np.array_equal([0, 1], difference):
            self.__pos = np.array(new_pos)
            if self.__opp == 1:
                self.__patch = Circle((self.__pos[0]+0.5, self.__pos[1]+0.5), radius = 0.25, fc = 'white')
            if self.__opp == 2:
                self.__patch = Circle((self.__pos[0]+0.5, self.__pos[1]+0.5), radius = 0.25, fc = 'black')
            self.__moves += 1
        else: 
            return False

    def take(self, new_pos):
        new_pos = np.array(new_pos)
        if self.__opp == 1:
            difference = [self.__pos[0]-new_pos[0], self.__pos[1]-new_pos[1]]
        elif self.__opp == 2:
            difference = [new_pos[0]-self.__pos[0], new_pos[1]-self.__pos[1]]
        av_diff = [[1, 1], [-1, 1]]
        if any(np.array_equal(difference, array) for array in av_diff) and 0 <= new_pos[0] <= 7 and 0 <= new_pos[1] <= 7:
            self.__pos = np.array(new_pos)
            self.__moves += 1
        else: 
            return None
        


    

class King():
    def __init__(self, opp, pos):
        self.__pos = np.array(pos)
        self.__moves = 0
        self.__opp = opp # 1 is white, 2 is black 
        if opp == 1:
            self.__patch = Rectangle(self.__pos, 0.5, 0.5, fc = 'white')
        elif opp == 2:
            self.__patch = Rectangle(self.__pos, 0.5, 0.5, fc = 'black')

    def pos(self):
        return self.__pos
    
    def opp(self):
        return self.__opp
    
    def patch(self):
        return self.__patch
    
    def move(self, new_pos):
        new_pos = np.array(new_pos)
        difference = [new_pos[0]-self.__pos[0], new_pos[1]-self.__pos[1]]
        print(difference)

        av_diff = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, 1]]
        if any(np.array_equal(difference, array) for array in av_diff):
                self.__pos = np.array(new_pos) 
                self.__moves += 1 
        else:
            return None
        
                
    def take(self, new_pos):
        self.move(new_pos)

    

        

        

    


