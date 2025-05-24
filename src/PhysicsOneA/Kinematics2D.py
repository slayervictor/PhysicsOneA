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
                 time_of_flight=None,
                 y0=0, g=gravity()):
        """
        Initializes a projectile motion object.

        You must provide one and only one of the following input combinations:

        1. v0 and theta (float, float):
            - v0: Initial speed (m/s)
            - theta: Launch angle (radians)

        2. v0x and v0y (float, float):
            - v0x: Horizontal velocity (m/s)
            - v0y: Vertical velocity (m/s)

        3. vector (Vector):
            - A 2D vector with [v0x, v0y]

        4. vector_pair (VectorPair):
            - Displacement between two vectors used as velocity

        5. theta and time_of_flight:
            - theta: Launch angle (radians)
            - time_of_flight: Total flight time in seconds

        Optional:
            - y0 (float): Launch height in meters (default: 0)
            - g (float): Gravitational acceleration in m/s² (default: gravity())

        Raises:
            ValueError: If valid input is not provided
            TypeError: If vector or vector_pair have invalid type
        """
        self.y0 = y0
        self.g = g

        # Method 1: Vector instance
        if vector is not None:
            if not isinstance(vector, Vector):
                raise TypeError("The vector must be an instance of the Vector class.")
            vec = vector.getVector()
            if len(vec) < 2:
                raise ValueError("Vector must have at least 2 components.")
            self.v0x = vec[0]
            self.v0y = vec[1]
            self.v0 = sqrt(self.v0x**2 + self.v0y**2)
            self.theta = atan2(self.v0y, self.v0x)

        # Method 2: VectorPair instance
        elif vector_pair is not None:
            if not isinstance(vector_pair, VectorPair):
                raise TypeError("vector_pair must be an instance of the VectorPair class.")
            disp = vector_pair.displacement()
            if len(disp) < 2:
                raise ValueError("VectorPair must be at least 2D.")
            self.v0x = disp[0]
            self.v0y = disp[1]
            self.v0 = sqrt(self.v0x**2 + self.v0y**2)
            self.theta = atan2(self.v0y, self.v0x)

        # Method 3: v0 and theta
        elif v0 is not None and theta is not None:
            self.v0 = v0
            self.theta = theta
            self.v0x = v0 * cos(theta)
            self.v0y = v0 * sin(theta)

        # Method 4: v0x and v0y
        elif v0x is not None and v0y is not None:
            self.v0x = v0x
            self.v0y = v0y
            self.v0 = sqrt(v0x**2 + v0y**2)
            self.theta = atan2(v0y, v0x)

        # Method 5: theta and total time of flight (solve for v0)
        elif theta is not None and time_of_flight is not None:
            if time_of_flight <= 0:
                raise ValueError("Flight time must be positive.")
            numerator = 0.5 * self.g * time_of_flight**2 - self.y0
            denominator = time_of_flight * sin(theta)
            if denominator == 0:
                raise ValueError("Cannot compute v0: sin(theta) is zero.")
            self.v0 = numerator / denominator
            self.theta = theta
            self.v0x = self.v0 * cos(theta)
            self.v0y = self.v0 * sin(theta)

        else:
            raise ValueError("You must provide either (v0 and theta), (v0x and v0y), a Vector, a VectorPair, or (theta and time_of_flight).")

    def __str__(self):
        return (
            f"--- Projectile Info ---\n"
            f"Initial speed (v0): {self.v0:.2f} m/s\n"
            f"Launch angle (theta): {degrees(self.theta):.2f}°\n"
            f"Horizontal velocity (v0x): {self.v0x:.2f} m/s\n"
            f"Vertical velocity (v0y): {self.v0y:.2f} m/s\n"
            f"Initial height (y0): {self.y0:.2f} m\n"
            f"Gravity (g): {self.g:.2f} m/s²\n"
            f"\n--- Derived Quantities ---\n"
            f"Time of flight (flat): {self.time_of_flight():.2f} s\n"
            f"Time of flight (realistic): {self.time_of_flight_full():.2f} s\n"
            f"Maximum height: {self.max_height():.2f} m\n"
            f"Range: {self.range():.2f} m"
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

    def position(self, t, allow_outside=False):
        """
        Returns the position (x, y) of the projectile at time t.

        Parameters:
            t (float): Time in seconds
            allow_outside (bool): If False, raise error if t is outside flight time

        Returns:
            tuple: (x, y) in meters

        Raises:
            ValueError: If t is outside [0, time_of_flight_full()] and allow_outside is False
        """
        t_max = self.time_of_flight_full()
        if not allow_outside and (t < 0 or t > t_max):
            raise ValueError(f"Time {t} s is outside of projectile's flight time [0, {t_max:.2f}]")

        return self.position(t)

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

    def plot_y_over_time(self, steps=100):
        t_max = self.time_of_flight_full()
        times = [i * t_max / steps for i in range(steps + 1)]
        ys = [self.position(t)[1] for t in times]

        plt.plot(times, ys)
        plt.xlabel("Time (s)")
        plt.ylabel("Height y(t) (m)")
        plt.title("Projectile Height Over Time")
        plt.grid()
        plt.show()

    def plot_x_over_time(self, steps=100):
        t_max = self.time_of_flight_full()
        times = [i * t_max / steps for i in range(steps + 1)]
        xs = [self.position(t)[0] for t in times]

        plt.plot(times, xs)
        plt.xlabel("Time (s)")
        plt.ylabel("Horizontal Distance x(t) (m)")
        plt.title("Projectile Horizontal Position Over Time")
        plt.grid()
        plt.show()

    def plot_velocity_over_time(self, steps=100):
        t_max = self.time_of_flight_full()
        times = [i * t_max / steps for i in range(steps + 1)]
        vxs = [self.velocity(t)[0] for t in times]
        vys = [self.velocity(t)[1] for t in times]

        plt.plot(times, vxs, label="vx(t)")
        plt.plot(times, vys, label="vy(t)")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Projectile Velocity Components Over Time")
        plt.legend()
        plt.grid()
        plt.show()


    def get_initial_vector(self):
        return Vector([self.v0x, self.v0y])
    
    # Position as a function of time (horizontal)
    def x(self, t):
        """
        Returns the horizontal position x(t) in meters at time t (seconds).
        """
        return self.v0x * t

    # Position as a function of time (vertical)
    def y(self, t):
        """
        Returns the vertical position y(t) in meters at time t (seconds), accounting for gravity.
        """
        return self.v0y * t - 0.5 * self.g * t**2 + self.y0

    # Horizontal velocity (constant)
    def vx(self, t):
        """
        Returns the horizontal velocity vx(t) in m/s, which is constant.
        """
        return self.v0x

    # Vertical velocity as a function of time
    def vy(self, t):
        """
        Returns the vertical velocity vy(t) in m/s at time t (seconds), decreasing due to gravity.
        """
        return self.v0y - self.g * t

    # Squared vertical velocity at a given height
    def vy_squared(self, y):
        """
        Returns vy² at a given vertical position y (meters), using energy conservation.
        """
        return self.v0y**2 - 2 * self.g * (y - self.y0)

    # Total speed squared at a given height
    def v_total_squared(self, y):
        """
        Returns total velocity squared v² at a given height y (meters), without using time.
        """
        return self.v0**2 - 2 * self.g * (y - self.y0)

    # Vertical position as a function of horizontal position
    def y_from_x(self, x):
        """
        Returns the vertical position y as a function of horizontal position x (meters).
        This is the trajectory equation y(x) for projectile motion.
        """
        return self.y0 + tan(self.theta) * x - (self.g / (2 * self.v0**2 * cos(self.theta)**2)) * x**2


