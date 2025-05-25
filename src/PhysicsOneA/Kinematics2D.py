from PhysicsOneA.dependencies import *
from PhysicsOneA.helpers import *
def is_ufloat(x):
    return hasattr(x, "nominal_value")

def smart_sqrt(x):
    return usqrt(x) if is_ufloat(x) else sqrt(x)

def smart_sin(x):
    return usin(x) if is_ufloat(x) else sin(x)

def smart_cos(x):
    return ucos(x) if is_ufloat(x) else cos(x)

def smart_tan(x):
    return utan(x) if is_ufloat(x) else tan(x)

def smart_atan2(y, x):
    return uatan2(y, x) if is_ufloat(y) or is_ufloat(x) else atan2(y, x)

class Vector:
    """
    Represents a mathematical vector that can include uncertainties in its components.
    Uses sympy.Matrix for internal representation and uncertainties.ufloat for error propagation.
    """
    def __init__(self, vec: Union[list, 'Vector']):
        """
        Initializes a Vector object.

        Args:
            vec (list or Vector): A list of numerical values or UFloat values, or another Vector.
        Raises:
            TypeError: If input is not a list or Vector.
        """
        if not isinstance(vec, list) and not isinstance(vec, Vector):
            raise TypeError("vec must be a list or Vector")
        if isinstance(vec, Vector):
            self.vec = vec.getVector()
        else:
            # Ensure all values are UFloat
            self.vec = Matrix([
                ufloat(val, 0) if not isinstance(val, UFloat) else val
                for val in vec
            ])

    def getVector(self):
        """
        Returns the underlying sympy.Matrix representing the vector.

        Returns:
            sympy.Matrix: The vector components.
        """
        return self.vec

    def length(self):
        """
        Calculates the Euclidean length (norm) of the vector with uncertainty propagation.

        Returns:
            UFloat: The length of the vector with uncertainty.
        """
        return sqrt(sum(v**2 for v in self.vec))

    def __str__(self):
        """
        Returns a string representation of the vector and its length.

        Returns:
            str: The formatted string.
        """
        return f"Vector: {self.vec}\nLength: {self.length()}"

class Time:
    """
    Represents a time interval, allowing uncertainty in the start and end times.
    """
    def __init__(self, _from: Union[float, UFloat], _to: Union[float, UFloat]):
        """
        Initializes a Time object.

        Args:
            _from (float or UFloat): The start time (can include uncertainty).
            _to (float or UFloat): The end time (can include uncertainty).
        """
        self.period = [
            ufloat(_from, 0) if not isinstance(_from, UFloat) else _from,
            ufloat(_to, 0) if not isinstance(_to, UFloat) else _to
        ]

    def delta_time(self):
        """
        Calculates the duration between start and end times with uncertainty.

        Returns:
            UFloat: The time difference with uncertainty.
        """
        return self.period[1] - self.period[0]

class Vector:
    """
    Simple wrapper class for a mathematical vector using sympy.Matrix.
    """

    def __init__(self, values):
        """
        Initializes the Vector with a list of values.

        Parameters:
            values (list): A list of numeric or ufloat elements representing a vector.
        """
        self.vec = Matrix(values)

    def getVector(self):
        """
        Returns the underlying vector (as a sympy Matrix).
        """
        return self.vec

    def __str__(self):
        """
        String representation of the vector.
        """
        return str(self.vec)

class VectorPair:
    """
    A class to represent a pair of vectors, useful for calculating displacement and direction angle.
    Supports vectors with uncertainties (ufloat).
    """

    def __init__(self, vec1, vec2):
        """
        Initializes the VectorPair with two vectors.

        Parameters:
            vec1 (list or Vector): The first vector (list of values or Vector object).
            vec2 (list or Vector): The second vector (list of values or Vector object).

        Raises:
            TypeError: If inputs are not lists or Vector instances.
        """
        if not isinstance(vec1, list) and not isinstance(vec1, Vector):
            raise TypeError("vec1 must be a list or a Vector")
        if not isinstance(vec2, list) and not isinstance(vec2, Vector):
            raise TypeError("vec2 must be a list or a Vector")

        self.vec1 = Matrix(vec1) if isinstance(vec1, list) else vec1.getVector()
        self.vec2 = Matrix(vec2) if isinstance(vec2, list) else vec2.getVector()
    
    def getPair(self):
        """
        Returns the pair of vectors as a tuple of sympy Matrices.

        Returns:
            tuple: (Vector object for vec1, Vector object for vec2)
        """
        return (Vector(list(self.vec1)).getVector(), Vector(list(self.vec2)).getVector())

    def getVector(self, index=None):
        """
        Returns one of the two vectors by index.

        Parameters:
            index (int): 0 for the first vector, 1 for the second.

        Returns:
            sympy.Matrix: The requested vector.

        Raises:
            ValueError: If index is not 0 or 1.
        """
        if index == 0:
            return Vector(list(self.vec1)).getVector()
        elif index == 1:
            return Vector(list(self.vec2)).getVector()
        else:
            raise ValueError("Index must be 0 or 1")

    def displacement(self):
        """
        Computes the displacement vector from vec1 to vec2.

        Returns:
            sympy.Matrix: The displacement vector (vec2 - vec1).
        """
        return self.vec2 - self.vec1

    def direction_angle(vector_pair):
        """
        Calculates the angle of displacement (in degrees) from vec1 to vec2 in the X-Y plane.

        If components have uncertainties (ufloat), the result will also carry uncertainty.

        Parameters:
            vector_pair (VectorPair): An instance of VectorPair.

        Returns:
            float or ufloat: The direction angle in degrees.

        Raises:
            IndexError: If the vectors are less than 2D.
        """
        disp = vector_pair.displacement()
        
        if len(disp) < 2:
            raise IndexError("Vectors must have at least two dimensions for angle calculation.")

        Vx = disp[0]
        Vy = disp[1]

        angle_rad = atan2(Vy, Vx)
        return angle_rad * 180 / pi

    def __str__(self):
        """
        Returns a string representation of the vector pair.
        """
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
            self.v0 = smart_sqrt(self.v0x**2 + self.v0y**2)
            self.theta = smart_atan2(self.v0y, self.v0x)

        elif vector_pair is not None:
            disp = vector_pair.displacement()
            self.v0x, self.v0y = disp[0], disp[1]
            self.v0 = smart_sqrt(self.v0x**2 + self.v0y**2)
            self.theta = smart_atan2(self.v0y, self.v0x)

        elif v0 is not None and theta is not None:
            self.v0 = v0
            self.theta = theta
            self.v0x = v0 * smart_cos(theta)
            self.v0y = v0 * smart_sin(theta)

        elif v0x is not None and v0y is not None:
            self.v0x = v0x
            self.v0y = v0y
            self.v0 = smart_sqrt(v0x**2 + v0y**2)
            self.theta = smart_atan2(v0y, v0x)

        elif theta is not None and time_of_flight is not None:
            numerator = 0.5 * self.g * time_of_flight**2 - self.y0
            denominator = time_of_flight * smart_sin(theta)
            self.v0 = numerator / denominator
            self.theta = theta
            self.v0x = self.v0 * smart_cos(theta)
            self.v0y = self.v0 * smart_sin(theta)

        elif max_height is not None and range_ is not None:
            best_theta = None
            min_error = float('inf')
            for deg in range(500, 860):
                th = radians(deg / 100)
                lhs = max_height / range_
                rhs = (smart_sin(th)**2) / (2 * smart_sin(2 * th))
                err = abs(lhs - rhs)
                if err < min_error:
                    best_theta = th
                    min_error = err
            self.theta = best_theta
            self.v0 = smart_sqrt((2 * self.g * max_height) / (smart_sin(self.theta)**2))
            self.v0x = self.v0 * smart_cos(self.theta)
            self.v0y = self.v0 * smart_sin(self.theta)

        elif range_ is not None and time_of_flight is not None:
            self.v0x = range_ / time_of_flight
            self.v0y = (self.g * time_of_flight) / 2
            self.v0 = smart_sqrt(self.v0x**2 + self.v0y**2)
            self.theta = smart_atan2(self.v0y, self.v0x)

        elif initial_point is not None and target_point is not None:
            dx = target_point[0] - initial_point[0]
            dy = target_point[1] - initial_point[1]
            self.theta = smart_atan2(dy, dx)
            self.v0 = smart_sqrt((self.g * dx**2) / (2 * smart_cos(self.theta)**2 * (dx * smart_tan(self.theta) - dy)))
            self.v0x = self.v0 * smart_cos(self.theta)
            self.v0y = self.v0 * smart_sin(self.theta)

        elif range_ is not None and impact_height is not None:
            best_theta = None
            min_error = float('inf')
            for deg in range(5, 86):
                th = radians(deg)
                try:
                    v0 = smart_sqrt((self.g * range_**2) / (2 * smart_cos(th)**2 * (range_ * smart_tan(th) + self.y0 - impact_height)))
                    y_calc = self.y0 + smart_tan(th) * range_ - (self.g * range_**2) / (2 * v0**2 * smart_cos(th)**2)
                    err = abs(y_calc - impact_height)
                    if err < min_error:
                        best_theta, best_v0 = th, v0
                        min_error = err
                except:
                    continue
            self.theta = best_theta
            self.v0 = best_v0
            self.v0x = self.v0 * smart_cos(self.theta)
            self.v0y = self.v0 * smart_sin(self.theta)

        elif apex_point is not None:
            h = apex_point[1] - self.y0
            x_half = apex_point[0]
            self.theta = smart_atan2(h, x_half)
            self.v0 = smart_sqrt((2 * self.g * h) / (smart_sin(self.theta)**2))
            self.v0x = self.v0 * smart_cos(self.theta)
            self.v0y = self.v0 * smart_sin(self.theta)

        elif start is not None and end is not None and max_height is not None:
            dx = end[0] - start[0]
            h = max_height - start[1]
            self.theta = smart_atan2(h, dx / 2)
            self.v0 = smart_sqrt((2 * self.g * h) / (smart_sin(self.theta)**2))
            self.v0x = self.v0 * smart_cos(self.theta)
            self.v0y = self.v0 * smart_sin(self.theta)
            self.y0 = start[1]

        else:
            raise ValueError("You must provide valid input combination to initialize the projectile.")
    def _format_value(self, value):
        """
        Format a numeric value with or without uncertainty for display.

        Parameters:
            value (float or UFloat): A numeric value, possibly with uncertainty.
                - Units: arbitrary (context-dependent)

        Returns:
            str: Formatted string of the value rounded to 4 decimal places.
        """
        if hasattr(value, 'nominal_value'):
            return f"{value:.4f}"
        return f"{float(value):.4f}"

    def __str__(self):
        """
        Return a formatted summary of the projectile's parameters and derived properties.

        Returns:
            str: A human-readable string listing:
                - Initial speed (m/s)
                - Launch angle (degrees)
                - Initial velocity components (v0x, v0y) in m/s
                - Initial height y0 (m)
                - Gravitational acceleration (m/s^2)
                - Time of flight (s)
                - Maximum height (m)
                - Range (m)
        """
        return (
            f"--- Projectile Info ---\n"
            f"Initial speed (v0): {self._format_value(self.v0)} m/s\n"
            f"Launch angle (theta): {round(N(radian_to_degree(self.theta)),2)}\u00b0\n"
            f"Horizontal velocity (v0x): {self._format_value(self.v0x)} m/s\n"
            f"Vertical velocity (v0y): {self._format_value(self.v0y)} m/s\n"
            f"Initial height (y0): {self._format_value(self.y0)} m\n"
            f"Gravity (g): {self.g} m/s²\n"
            f"\n--- Derived Quantities ---\n"
            f"Time of flight (flat): {self._format_value(self.time_of_flight())} s\n"
            f"Time of flight (realistic): {self._format_value(self.time_of_flight_full())} s\n"
            f"Maximum height: {self._format_value(self.max_height())} m\n"
            f"Range: {self._format_value(self.range())} m"
        )

    def time_of_flight(self):
        """
        Calculate the time of flight assuming the projectile lands at the initial height.

        Returns:
            float or UFloat: Time in seconds (s) until the projectile returns to height y0.
        """
        return (2 * self.v0y) / self.g

    def time_of_flight_full(self):
        """
        Calculate the full flight duration until the projectile hits the ground (y=0).

        Returns:
            float or UFloat: Time in seconds (s) from launch to ground impact.
        """
        a = -0.5 * self.g
        b = self.v0y
        c = self.y0
        discriminant = b**2 - 4 * a * c
        sqrt_disc = usqrt(discriminant) if hasattr(discriminant, "nominal_value") else sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        if hasattr(t1, "nominal_value"):
            return t1 if t1.nominal_value > t2.nominal_value else t2
        return max(t1, t2)

    def max_height(self):
        """
        Compute the maximum vertical height reached by the projectile.

        Returns:
            float or UFloat: Maximum height in meters (m).
        """
        return self.y0 + (self.v0y ** 2) / (2 * self.g)

    def range(self):
        """
        Calculate the horizontal distance traveled until the projectile lands.

        Returns:
            float or UFloat: Horizontal range in meters (m).
        """
        return (self.v0 ** 2) * smart_sin(2 * self.theta) / self.g

    def position(self, t, allow_outside=False):
        """
        Get the position (x, y) of the projectile at time t.

        Parameters:
            t (float): Time since launch in seconds (s).
            allow_outside (bool): If False, restricts t to within the flight time.

        Returns:
            Tuple[float or UFloat, float or UFloat]: Position (x, y) in meters (m).
        """
        t_max = self.time_of_flight_full()
        t_max_val = t_max.nominal_value if hasattr(t_max, 'nominal_value') else t_max
        if not allow_outside and (t < 0 or t > t_max_val):
            raise ValueError(f"Time {t} s is outside of projectile's flight time [0, {t_max_val:.2f}]")
        return (self.x(t), self.y(t))

    def trajectory_y(self, x):
        """
        Calculate the vertical position y as a function of horizontal position x.

        Parameters:
            x (float): Horizontal distance in meters (m).

        Returns:
            float or UFloat: Vertical position y(x) in meters (m).
        """
        return self.y0 + smart_tan(self.theta) * x - (self.g / (2 * self.v0 ** 2 * smart_cos(self.theta) ** 2)) * x ** 2

    def plot_trajectory(self, steps=100):
        """
        Plot the projectile's 2D trajectory (x vs y).

        Parameters:
            steps (int): Number of sample points used to compute the curve.
        """
        t_max = self.time_of_flight()
        t_max_val = t_max.nominal_value if hasattr(t_max, 'nominal_value') else t_max
        times = [i * t_max_val / steps for i in range(steps + 1)]
        positions = [self.position(t) for t in times]
        xs = [pos[0].nominal_value if hasattr(pos[0], 'nominal_value') else pos[0] for pos in positions]
        ys = [pos[1].nominal_value if hasattr(pos[1], 'nominal_value') else pos[1] for pos in positions]

        plt.plot(xs, ys)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Projectile Trajectory")
        plt.grid()
        plt.show()

    def plot_y_over_time(self, steps=100):
        """
        Plot vertical position y(t) over time.

        Parameters:
            steps (int): Number of time samples.
        """
        t_max = self.time_of_flight_full()
        t_max_val = t_max.nominal_value if hasattr(t_max, 'nominal_value') else t_max
        times = [i * t_max_val / steps for i in range(steps + 1)]
        ys = [self.position(t)[1] for t in times]
        ys_vals = [y.nominal_value if hasattr(y, 'nominal_value') else y for y in ys]

        plt.plot(times, ys_vals)
        plt.xlabel("Time (s)")
        plt.ylabel("Height y(t) (m)")
        plt.title("Projectile Height Over Time")
        plt.grid()
        plt.show()

    def plot_x_over_time(self, steps=100):
        """
        Plot horizontal position x(t) over time.

        Parameters:
            steps (int): Number of time samples.
        """
        t_max = self.time_of_flight_full()
        t_max_val = t_max.nominal_value if hasattr(t_max, 'nominal_value') else t_max
        times = [i * t_max_val / steps for i in range(steps + 1)]
        xs = [self.position(t)[0] for t in times]
        xs_vals = [x.nominal_value if hasattr(x, 'nominal_value') else x for x in xs]

        plt.plot(times, xs_vals)
        plt.xlabel("Time (s)")
        plt.ylabel("Horizontal Distance x(t) (m)")
        plt.title("Projectile Horizontal Position Over Time")
        plt.grid()
        plt.show()

    def plot_velocity_over_time(self, steps=100):
        """
        Plot the velocity components vx(t) and vy(t) over time.

        Parameters:
            steps (int): Number of time samples.
        """
        t_max = self.time_of_flight_full()
        t_max_val = t_max.nominal_value if hasattr(t_max, 'nominal_value') else t_max
        times = [i * t_max_val / steps for i in range(steps + 1)]
        velocities = [self.velocity(t) for t in times]
        vxs = [v[0].nominal_value if hasattr(v[0], 'nominal_value') else v[0] for v in velocities]
        vys = [v[1].nominal_value if hasattr(v[1], 'nominal_value') else v[1] for v in velocities]

        plt.plot(times, vxs, label="vx(t)")
        plt.plot(times, vys, label="vy(t)")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Projectile Velocity Components Over Time")
        plt.legend()
        plt.grid()
        plt.show()

    def get_initial_vector(self):
        """
        Return the initial velocity vector [v0x, v0y].

        Returns:
            Vector: The initial velocity vector (m/s).
        """
        return Vector([self.v0x, self.v0y])

    def x(self, t):
        """
        Compute horizontal position x at time t.

        Parameters:
            t (float): Time in seconds (s).

        Returns:
            float or UFloat: Horizontal position x(t) in meters (m).
        """
        return self.v0x * t

    def y(self, t):
        """
        Compute vertical position y at time t.

        Parameters:
            t (float): Time in seconds (s).

        Returns:
            float or UFloat: Vertical position y(t) in meters (m).
        """
        return self.v0y * t - 0.5 * self.g * t**2 + self.y0

    def vx(self, t):
        """
        Get horizontal velocity component (constant).

        Parameters:
            t (float): Time (ignored, included for consistency).

        Returns:
            float or UFloat: Horizontal velocity vx in m/s.
        """
        return self.v0x

    def vy(self, t):
        """
        Compute vertical velocity at time t.

        Parameters:
            t (float): Time in seconds (s).

        Returns:
            float or UFloat: Vertical velocity vy(t) in m/s.
        """
        return self.v0y - self.g * t

    def vy_squared(self, y):
        """
        Compute squared vertical velocity at a specific height.

        Parameters:
            y (float): Vertical position in meters (m).

        Returns:
            float or UFloat: vy^2 at height y (m^2/s^2).
        """
        return self.v0y**2 - 2 * self.g * (y - self.y0)

    def v_total_squared(self, y):
        """
        Compute total speed squared at a specific height.

        Parameters:
            y (float): Vertical position in meters (m).

        Returns:
            float or UFloat: v^2 at height y (m^2/s^2).
        """
        return self.v0**2 - 2 * self.g * (y - self.y0)

    def y_from_x(self, x):
        """
        Calculate vertical position y based on horizontal distance x using the trajectory equation.

        Parameters:
            x (float): Horizontal distance in meters (m).

        Returns:
            float or UFloat: Vertical position y in meters (m).
        """
        return self.y0 + smart_tan(self.theta) * x - (self.g / (2 * self.v0**2 * smart_cos(self.theta)**2)) * x**2

class CircularMotion:
    def __init__(self, *, r: Union[float, UFloat] = None, T: Union[float, UFloat] = None, v: Union[float, UFloat] = None, v_func=None):
        """
        Circular motion handler that supports both exact and uncertainty-aware values.

        Parameters:
            r (float or UFloat, optional): Radius of the circular path [m].
            T (float or UFloat, optional): Period of one revolution [s].
            v (float or UFloat, optional): Tangential velocity [m/s]. If not provided, computed using r and T.
            v_func (sympy expression, optional): Time-dependent velocity function v(t) [m/s].

        Notes:
            You can provide:
                - r and T: computes v automatically.
                - v and r: computes centripetal acceleration.
                - v_func: computes symbolic tangential acceleration over time.
        """
        self.r = r
        self.T = T
        self.v = v
        self.v_func = v_func
        self.t = symbols('t')

        if self.v is None:
            if self.r is not None and self.T is not None:
                self.v = (2 * pi * self.r) / self.T
            elif self.r is None and self.T is not None:
                raise ValueError("Cannot compute velocity: radius (r) is missing.")
            elif self.r is not None and self.T is None:
                raise ValueError("Cannot compute velocity: period (T) is missing.")

    def velocity(self) -> Union[float, UFloat]:
        """
        Returns the tangential velocity v.

        Formula:
            v = (2πr) / T

        Returns:
            float or UFloat: Tangential velocity [m/s]
        """
        if self.v is not None:
            return self.v
        elif self.r is not None and self.T is not None:
            return (2 * pi * self.r) / self.T
        else:
            raise ValueError("Missing values: Provide either v, or both r and T.")

    def centripetal_acceleration(self) -> Union[float, UFloat]:
        """
        Returns the centripetal acceleration a_c.

        Formula:
            a_c = v² / r

        Returns:
            float or UFloat: Centripetal acceleration [m/s²]

        Raises:
            ValueError: If radius r is not provided.
        """
        v = self.velocity()
        if self.r is None:
            raise ValueError("Radius r is required to compute centripetal acceleration.")
        return (v**2) / self.r

    def tangential_acceleration(self):
        """
        Returns the symbolic expression for tangential acceleration a_T.

        Formula:
            a_T = d|v(t)| / dt

        Returns:
            sympy expression: Tangential acceleration [m/s²]

        Raises:
            ValueError: If no symbolic v_func is provided.

        Example:
            t = symbols('t')
            v_func = 5 * sin(t)
        """
        if self.v_func is None:
            raise ValueError("A time-dependent velocity function v(t) is required for tangential acceleration.")
        return diff(abs(self.v_func), self.t)

    def critical_period_for_weightlessness(self, g: Union[float, UFloat] = gravity()) -> Union[float, UFloat]:
        """
        Computes the critical period T at which centripetal acceleration equals gravity (weightlessness).

        Formula:
            T = sqrt((4π²r) / g)

        Parameters:
            g (float or UFloat): Gravitational acceleration [m/s²]

        Returns:
            float or UFloat: Critical period T [s]

        Raises:
            ValueError: If radius r is not provided.
        """
        if self.r is None:
            raise ValueError("Radius r is required to compute the critical period.")

        try:
            if isinstance(self.r, UFloat) or isinstance(g, UFloat):
                return usqrt((4 * pi**2 * self.r) / g)
        except Exception:
            pass
        return sqrt((4 * pi**2 * self.r) / g)

    def __str__(self):
        """
        Returns a string representation of the circular motion object's parameters and results.
        """
        try:
            v_val = self.velocity()
            v_str = f"v = {v_val:.4f} m/s"
        except:
            v_str = "v = ?"

        try:
            ac_val = self.centripetal_acceleration()
            ac_str = f"centripetal acceleration = {ac_val:.4f} m/s²"
        except:
            ac_str = "centripetal acceleration = ?"

        try:
            at_val = self.tangential_acceleration()
            at_str = f"tangential acceleration = {at_val} m/s²"
        except:
            at_str = "tangential acceleration = ?"

        try:
            T_crit = self.critical_period_for_weightlessness()
            T_crit_hours = float(nominal_value(T_crit)) / 3600
            crit_str = f"Critical period for weightlessness: {T_crit:.2f} seconds (~{T_crit_hours:.2f} hours)"
        except:
            crit_str = "Critical period for weightlessness: ?"

        return (
            "--- Circular Motion ---\n"
            f"Radius r: {self.r} m\n"
            f"Period T: {self.T} s\n"
            f"{v_str}\n"
            f"{ac_str}\n"
            f"{at_str}\n"
            f"{crit_str}"
        )

class RelativeMotion:
    """
    Models relative motion between frames of reference using velocity and acceleration vectors.

    This class supports values with or without uncertainty (using `UFloat`).

    Parameters (all optional, keyword-only):
        v_ps_prime (Vector | list): Velocity of P relative to S′ [m/s].
        v_sprime_s (Vector | list): Velocity of S′ relative to S [m/s].
        a_ps_prime (Vector | list): Acceleration of P relative to S′ [m/s²].
        a_sprime_s (Vector | list): Acceleration of S′ relative to S [m/s²].
        velocity_vectors (tuple): Tuple of two velocity vectors [v_ps_prime, v_sprime_s].
        acceleration_vectors (tuple): Tuple of two acceleration vectors [a_ps_prime, a_sprime_s].
    
    Methods:
        relative_velocity(): Computes relative velocity vector [m/s].
        relative_acceleration(): Computes relative acceleration vector [m/s²].
    """

    def __init__(self, *,
                 v_ps_prime=None, v_sprime_s=None,
                 a_ps_prime=None, a_sprime_s=None,
                 velocity_vectors=None,
                 acceleration_vectors=None):
        
        def ensure_vector(v):
            if isinstance(v, Vector):
                return v
            elif isinstance(v, list):
                return Vector(v)
            else:
                raise TypeError("Expected a Vector or list.")

        if velocity_vectors:
            self.v_ps_prime = ensure_vector(velocity_vectors[0])
            self.v_sprime_s = ensure_vector(velocity_vectors[1])
        else:
            self.v_ps_prime = ensure_vector(v_ps_prime) if v_ps_prime is not None else None
            self.v_sprime_s = ensure_vector(v_sprime_s) if v_sprime_s is not None else None

        if acceleration_vectors:
            self.a_ps_prime = ensure_vector(acceleration_vectors[0])
            self.a_sprime_s = ensure_vector(acceleration_vectors[1])
        else:
            self.a_ps_prime = ensure_vector(a_ps_prime) if a_ps_prime is not None else None
            self.a_sprime_s = ensure_vector(a_sprime_s) if a_sprime_s is not None else None

    def relative_velocity(self) -> Vector:
        """
        Computes the total velocity of point P relative to reference frame S.

        Formula:
            v_PS = v_PS′ + v_S′S

        Returns:
            Vector: Relative velocity vector [m/s].

        Raises:
            ValueError: If required velocity vectors are not provided.
        """
        if self.v_ps_prime is None or self.v_sprime_s is None:
            raise ValueError("Both v_ps_prime and v_sprime_s must be provided.")
        return self.v_ps_prime + self.v_sprime_s

    def relative_acceleration(self) -> Vector:
        """
        Computes the total acceleration of point P relative to reference frame S.

        Formula:
            a_PS = a_PS′ + a_S′S

        Returns:
            Vector: Relative acceleration vector [m/s²].

        Raises:
            ValueError: If required acceleration vectors are not provided.
        """
        if self.a_ps_prime is None or self.a_sprime_s is None:
            raise ValueError("Both a_ps_prime and a_sprime_s must be provided.")
        return self.a_ps_prime + self.a_sprime_s

    def __str__(self) -> str:
        """
        Returns:
            str: Human-readable summary of velocity and acceleration results.
        """
        lines = ["--- Relative Motion ---"]
        if self.v_ps_prime and self.v_sprime_s:
            v_result = self.relative_velocity()
            lines += [
                f"v_PS′ = {self.v_ps_prime}",
                f"v_S′S = {self.v_sprime_s}",
                f"=> v_PS = {v_result}"
            ]
        else:
            lines.append("Velocity data incomplete.")

        if self.a_ps_prime and self.a_sprime_s:
            a_result = self.relative_acceleration()
            lines += [
                f"a_PS′ = {self.a_ps_prime}",
                f"a_S′S = {self.a_sprime_s}",
                f"=> a_PS = {a_result}"
            ]
        else:
            lines.append("Acceleration data incomplete.")
        return "\n".join(lines)