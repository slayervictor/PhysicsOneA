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
                 max_height=None,
                 range_=None,
                 impact_height=None,
                 initial_point=None,
                 target_point=None,
                 apex_point=None,
                 start=None,
                 end=None,
                 y0=0, g=gravity()):
        """
        Initializes a projectile motion object.

        Possible input combinations (provide only one):

        1. v0 (float, m/s) and theta (float, radians)
        2. v0x (float, m/s) and v0y (float, m/s)
        3. vector (Vector): 2D velocity vector
        4. vector_pair (VectorPair): initial displacement
        5. theta (float, radians) and time_of_flight (float, seconds)
        6. max_height (float, meters) and range_ (float, meters)
        7. range_ (float, meters) and flight_time (float, seconds)
        8. initial_point and target_point: each [x, y] (float, meters)
        9. range_ (float, meters) and impact_height (float, meters)
        10. apex_point: [x, y] (float, meters)
        11. start, end (each [x, y]) and max_height (float, meters)

        Optional:
            y0 (float): launch height (m)
            g (float): gravity (m/s^2)

        Automatically computes:
            v0, theta, v0x, v0y
        """
        self.y0 = y0
        self.g = g

        if vector is not None:
            vec = vector.getVector()
            self.v0x, self.v0y = vec[0], vec[1]
            self.v0 = sqrt(self.v0x**2 + self.v0y**2)
            self.theta = atan2(self.v0y, self.v0x)

        elif vector_pair is not None:
            disp = vector_pair.displacement()
            self.v0x, self.v0y = disp[0], disp[1]
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

        elif theta is not None and time_of_flight is not None:
            numerator = 0.5 * self.g * time_of_flight**2 - self.y0
            denominator = time_of_flight * sin(theta)
            self.v0 = numerator / denominator
            self.theta = theta
            self.v0x = self.v0 * cos(theta)
            self.v0y = self.v0 * sin(theta)

        elif max_height is not None and range_ is not None:
            best_theta = None
            min_error = float('inf')
            for deg in range(500, 860):  # 5.00° to 8.59°
                th = radians(deg / 100)
                lhs = max_height / range_
                rhs = (sin(th)**2) / (2 * sin(2 * th))
                err = abs(lhs - rhs)
                if err < min_error:
                    best_theta = th
                    min_error = err
            self.theta = best_theta
            self.v0 = sqrt((2 * self.g * max_height) / (sin(self.theta)**2))
            self.v0x = self.v0 * cos(self.theta)
            self.v0y = self.v0 * sin(self.theta)

        elif range_ is not None and time_of_flight is not None:
            self.v0x = range_ / time_of_flight
            self.v0y = (self.g * time_of_flight) / 2
            self.v0 = sqrt(self.v0x**2 + self.v0y**2)
            self.theta = atan2(self.v0y, self.v0x)

        elif initial_point is not None and target_point is not None:
            dx = target_point[0] - initial_point[0]
            dy = target_point[1] - initial_point[1]
            self.theta = atan2(dy, dx)
            # Assume flat arc at midpoint: v0 from dx and dy under gravity
            self.v0 = sqrt((self.g * dx**2) / (2 * cos(self.theta)**2 * (dx * tan(self.theta) - dy)))
            self.v0x = self.v0 * cos(self.theta)
            self.v0y = self.v0 * sin(self.theta)

        elif range_ is not None and impact_height is not None:
            # Assume symmetric arc shape from y0 to impact_height
            best_theta = None
            min_error = float('inf')
            for deg in range(5, 86):
                th = radians(deg)
                try:
                    v0 = sqrt((self.g * range_**2) / (2 * cos(th)**2 * (range_ * tan(th) + self.y0 - impact_height)))
                    y_calc = self.y0 + tan(th) * range_ - (self.g * range_**2) / (2 * v0**2 * cos(th)**2)
                    err = abs(y_calc - impact_height)
                    if err < min_error:
                        best_theta, best_v0 = th, v0
                        min_error = err
                except:
                    continue
            self.theta = best_theta
            self.v0 = best_v0
            self.v0x = self.v0 * cos(self.theta)
            self.v0y = self.v0 * sin(self.theta)

        elif apex_point is not None:
            # Max height and x from apex
            h = apex_point[1] - self.y0
            x_half = apex_point[0]
            self.theta = atan2(h, x_half)
            self.v0 = sqrt((2 * self.g * h) / (sin(self.theta)**2))
            self.v0x = self.v0 * cos(self.theta)
            self.v0y = self.v0 * sin(self.theta)

        elif start is not None and end is not None and max_height is not None:
            dx = end[0] - start[0]
            h = max_height - start[1]
            self.theta = atan2(h, dx / 2)
            self.v0 = sqrt((2 * self.g * h) / (sin(self.theta)**2))
            self.v0x = self.v0 * cos(self.theta)
            self.v0y = self.v0 * sin(self.theta)
            self.y0 = start[1]

        else:
            raise ValueError("You must provide valid input combination to initialize the projectile.")


    def __str__(self):
        return (
            f"--- Projectile Info ---\n"
            f"Initial speed (v0): {self.v0:.4f} m/s\n"
            f"Launch angle (theta): {round(N(radian_to_degree(self.theta)),2)}°\n"
            f"Horizontal velocity (v0x): {self.v0x:.4f} m/s\n"
            f"Vertical velocity (v0y): {self.v0y:.4f} m/s\n"
            f"Initial height (y0): {self.y0:.4f} m\n"
            f"Gravity (g): {self.g} m/s²\n"
            f"\n--- Derived Quantities ---\n"
            f"Time of flight (flat): {self.time_of_flight():.4f} s\n"
            f"Time of flight (realistic): {self.time_of_flight_full():.4f} s\n"
            f"Maximum height: {self.max_height():.4f} m\n"
            f"Range: {self.range():.4f} m"
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

class CircularMotion:
    def __init__(self, *, r=None, T=None, v=None, v_func=None):
        """
        Circular motion handler.

        Parameters:
            r (float or sympy expression): The radius of the circular path [in meters].
            T (float or sympy expression): The period — the time it takes to complete one full revolution [in seconds].
            v (float or sympy expression): The tangential (linear) velocity of the object [in meters per second].
            v_func (sympy expression): A symbolic expression representing the velocity as a function of time, v(t).
                                       Required to compute tangential acceleration a_T = d|v|/dt.

        You can provide the following combinations:

        - r and T: Used to compute the tangential velocity using the formula:
              v = (2πr) / T

        - v and r: Used to compute the centripetal acceleration using the formula:
              a_c = v² / r

        - v_func: A function of time, used to compute the tangential acceleration:
              a_T = d|v(t)| / dt

        Notes:
            - If only r and T are given, v is computed automatically.
            - If v is given directly, it will be used instead of computing it from r and T.
            - You can leave v_func out unless you're specifically modeling acceleration with time-varying speed.
        """
        self.r = r
        self.T = T
        self.v = v
        self.v_func = v_func
        self.t = symbols('t')

        # Automatically compute v if r and T are provided
        if self.v is None and self.r is not None and self.T is not None:
            self.v = Float((2 * pi * self.r) / self.T)

    def velocity(self):
        """
        Returns the velocity v = (2πr)/T if not directly provided.
        """
        if self.v is not None:
            return self.v
        elif self.r is not None and self.T is not None:
            return Float((2 * pi * self.r) / self.T)
        else:
            raise ValueError("Missing values: Provide either v, or both r and T.")

    def centripetal_acceleration(self):
        """
        Returns the centripetal acceleration: a_c = v² / r
        """
        v = self.velocity()
        if self.r is None:
            raise ValueError("Radius r is required to compute centripetal acceleration.")
        return Float(v**2 / self.r)

    def tangential_acceleration(self):
        """
        Returns the tangential acceleration: a_T = d|v|/dt, for a given v(t).
        """
        if self.v_func is None:
            raise ValueError("A time-dependent velocity function v(t) is required for tangential acceleration.")
        return diff(abs(self.v_func), self.t)

    def __str__(self):
        v_str = f"v = {self.velocity()} m/s" if self.v or (self.r and self.T) else "v = ?"
        ac_str = f"a_c = {self.centripetal_acceleration()} m/s²" if self.r else "a_c = ?"
        at_str = f"a_T = {self.tangential_acceleration()} m/s²" if self.v_func else "a_T = ?"

        return f"--- Circular Motion ---\n{v_str}\n{ac_str}\n{at_str}"
