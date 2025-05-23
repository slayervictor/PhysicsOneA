from Lectures.dependencies import *

class Vector:
    def __init__(self, vec: List):
        if not isinstance(vec, list):
            raise TypeError("vec must be a list")
        self.vec = vec
    
    def length(self):
        sumsquare = sum(i**2 for i in self.vec)
        return sqrt(sumsquare)

class VectorPair:
    def __init__(self, vec1,vec2):
        if not isinstance(vec1,list) or not isinstance(vec2,list):
            raise TypeError("vec must be a list")
        self.pair = [vec1,vec2]

def evalf(n):
    return N(n)