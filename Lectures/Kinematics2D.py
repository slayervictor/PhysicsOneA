from Lectures.dependencies import *

class Vector:
    def __init__(self, vec: List):
        if not isinstance(vec, list):
            raise TypeError("vec must be a list")
        self.vec = vec
    
    def length(self):
        sumsquare = sum(i**2 for i in self.vec)
        return sqrt(sumsquare)
    
    def __str__(self):
        return f"Vectors: {self.vec}\nLength: {self.length()}"

class VectorPair:
    def __init__(self, vec1,vec2):
        if not isinstance(vec1,list) and not isinstance(vec2,list):
            raise TypeError("vec must be a list")
        self.pair = [vec1,vec2]