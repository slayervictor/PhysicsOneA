from PhysicsOneA.dependencies import *
from PhysicsOneA.helpers import *

class Vector:
    def __init__(self, vec):
        if not isinstance(vec, list) and not isinstance(vec, Vector):
            raise TypeError("vec must be a list")
        if isinstance(vec,Vector):
            self.vec = vec.getVector()
        else:
            self.vec = Matrix(vec)
    
    def getVector(self):
        return self.vec

    def length(self):
        return self.vec.norm()
    
    def __str__(self):
        return f"Vector: {self.vec}\nLength: {self.length()}"

class Time:
    def __init__(self,_from: Float, _to: Float):
        self.period = [_from,_to]
    
    def delta_time(self):
        return self.period[1]-self.period[0]

class VectorPair:
    def __init__(self, vec1, vec2):
        if not isinstance(vec1, list) and not isinstance(vec1, Vector) or not isinstance(vec2, list) and not isinstance(vec2, Vector):
            raise TypeError("vec1 and vec2 must be lists")
        if isinstance(vec1,list):
            self.vec1 = Matrix(vec1)
        else:
            self.vec1 = vec1.getVector()
        if isinstance(vec2,list):
            self.vec2 = Matrix(vec2)
        else:
            self.vec2 = vec2.getVector()
    
    def getPair(self):
        return (Vector(list(self.vec1)).getVector(),Vector(list(self.vec2)).getVector())

    def getVector(self,index=None):
        if index == 0:
            return Vector(list(self.vec1)).getVector()
        elif index == 1:
            return Vector(list(self.vec2)).getVector()

    def displacement(self):
        return self.vec2 - self.vec1

    def direction_angle(vector_pair):
        """
        Calculates the direction angle (α) in degrees between two vectors
        using the arctangent of the displacement's y and x components.

        This angle represents the orientation of the displacement vector 
        from the first vector to the second in 2D space, based on:
            α = tan⁻¹(Vy / Vx)

        Parameters:
            vector_pair (VectorPair): An instance of the VectorPair class
                                    containing two vectors.

        Returns:
            float: The direction angle α in degrees.

        Raises:
            IndexError: If vectors are not at least 2-dimensional.
            ZeroDivisionError: If the x-component of the displacement is zero.
        """
        # Get displacement vector
        disp = vector_pair.displacement()
        
        # Ensure vector has at least two dimensions
        if len(disp) < 2:
            raise IndexError("Vectors must have at least two dimensions for angle calculation.")

        Vx = disp[0]
        Vy = disp[1]

        # Compute angle in radians using arctangent
        angle_rad = atan(Vy / Vx)

        # Convert to degrees for readability
        return degrees(N(angle_rad))

    def __str__(self):
        pair = self.getPair()
        return f"Vector Pair:\n  First: {pair[0]}\n  Second: {pair[1]}"

class Projectile:
    def __init__(self, *, 
                 v0=None, theta=None, 
                 v0x=None, v0y=None, 
                 vector=None, 
                 vector_pair=None,
                 y0=0, g=gravity()):
        """
    Initializes a projectile motion object.

    You must provide one and only one of the following input combinations:

    1. v0 and theta (float, float):
        - v0: Initial speed (magnitude of velocity) in m/s
        - theta: Launch angle in radians

    2. v0x and v0y (float, float):
        - v0x: Horizontal component of velocity in m/s
        - v0y: Vertical component of velocity in m/s

    3. vector (Vector):
        - A 2D vector representing the initial velocity (x and y components)

    4. vector_pair (VectorPair):
        - A pair of position vectors; the displacement is used as initial velocity

    Optional:
        - y0 (float): Initial launch height in meters (default: 0)
        - g (float): Gravitational acceleration in m/s² (default: gravity())

    Raises:
        ValueError: If no valid input combination is provided
        TypeError: If vector or vector_pair are not of the correct type
        ValueError: If a vector has less than 2 components

    Sets:
        - self.v0: Initial speed (float)
        - self.theta: Launch angle in radians (float)
        - self.v0x: Horizontal velocity (float)
        - self.v0y: Vertical velocity (float)
        - self.y0: Initial height (float)
        - self.g: Gravitational acceleration (float)
    """
        self.y0 = y0
        self.g = g

        if vector is not None:
            if not isinstance(vector, Vector):
                raise TypeError("The vector needs to be a instance of Vector-class")
            vec = vector.getVector()
            if len(vec) < 2:
                raise ValueError("Vector needs at minimum 2 components")
            self.v0x = vec[0]
            self.v0y = vec[1]
            self.v0 = sqrt(self.v0x**2 + self.v0y**2)
            self.theta = atan2(self.v0y, self.v0x)

        elif vector_pair is not None:
            if not isinstance(vector_pair, VectorPair):
                raise TypeError("vector_pair needs to be an instance of the VectorPair-class")
            disp = vector_pair.displacement()
            if len(disp) < 2:
                raise ValueError("VectorPair needs a 2D-difference")
            self.v0x = disp[0]
            self.v0y = disp[1]
            self.v0 = sqrt(self.v0x**2 + self.v0y**2)
            self.theta = atan2(self.v0y, self.v0x)

        elif v0 is not None and theta is not None:
            self.v0 = v0
            self.theta = theta
            self.v0x = v0 * cos(theta)
            self.v0y = v0 * sin(theta)

        elif v0x is not None and v0y is not None:
            self.v0x = v0x
            self.v0y = v0y
            self.v0 = sqrt(v0x**2 + v0y**2)
            self.theta = atan2(v0y, v0x)

        else:
            raise ValueError("You need to give either (v0 og theta), (v0x og v0y), a Vector, or a VectorPair")

    def __str__(self):
        return (
            f"v0: {self.v0:.2f} m/s\n"
            f"theta: {degrees(self.theta):.2f}° (degrees)\n"
            f"v0x: {self.v0x:.2f} m/s\n"
            f"v0y: {self.v0y:.2f} m/s\n"
            f"y0: {self.y0:.2f} m\n"
            f"g: {self.g:.2f} m/s²\n"
            f"time of flight: {self.time_of_flight():.2f} s"
        )


    def time_of_flight(self):
        return (2 * self.v0y) / self.g
    
    def time_of_flight_full(self):
        """
        Solves 0 = y0 + v0y * t - 0.5 * g * t²
        Returns the positive root.
        """
        a = -0.5 * self.g
        b = self.v0y
        c = self.y0
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            raise ValueError("Projectile never reaches ground level")

        t1 = (-b + sqrt(discriminant)) / (2 * a)
        t2 = (-b - sqrt(discriminant)) / (2 * a)
        return max(t1, t2)


    def max_height(self):
        return self.y0 + (self.v0y ** 2) / (2 * self.g)
    
    def range(self):
        return (self.v0 ** 2) * sin(2 * self.theta) / self.g

    def position(self, t):
        x = self.v0x * t
        y = self.y0 + self.v0y * t - 0.5 * self.g * t ** 2
        return (x, y)

    def velocity(self, t):
        vx = self.v0x
        vy = self.v0y - self.g * t
        return (vx, vy)

    def trajectory_y(self, x):
        """Calculates y as a function of x (without time)."""
        return self.y0 + tan(self.theta) * x - (self.g / (2 * self.v0 ** 2 * cos(self.theta) ** 2)) * x ** 2
    
    def plot_trajectory(self, steps=100):
        t_max = self.time_of_flight()
        times = [i * t_max / steps for i in range(steps + 1)]
        xs = [self.position(t)[0] for t in times]
        ys = [self.position(t)[1] for t in times]

        plt.plot(xs, ys)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Projectile Trajectory")
        plt.grid()
        plt.show()

    def get_initial_vector(self):
        return Vector([self.v0x, self.v0y])
