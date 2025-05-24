from PhysicsOneA.dependencies import *

def speed_converter(value, from_unit='m/s', to_unit='km/h'):
   """
   Convert between km/s, m/h, m/s and km/h.

   Default (no args): m/s to km/h
   
   Parameters:
   value: numeric value to convert
   from_unit: 'm/s' or 'km/h' 
   to_unit: 'm/s' or 'km/h'
   
   Returns: converted value
   """
   if from_unit == to_unit:
       return value
   
   if from_unit == 'm/s' and to_unit == 'km/h':
       return value * 3.6
   elif from_unit == 'km/h' and to_unit == 'm/s':
       return value / 3.6
   else:
       raise ValueError("Units must be 'm/s' or 'km/h'")
   
def gravity(decimalPoints=2):
    return round(9.816012123456, decimalPoints)

def radian_to_degree(angle_rad):
    return deg(angle_rad)

def degree_to_radian(angle_deg):
    return rad(angle_deg)

def calculate_from_angle(angle_deg: float):
    """
    Calculates sin, cos, and tan from a given angle in degrees using symbolic math (sympy).
    Also converts angle to radians for math accuracy.
    """
    angle_rad = radians(angle_deg)
    angle_sym = symbols('θ')
    
    sin_val = sin(rad(angle_deg))
    cos_val = cos(rad(angle_deg))
    tan_val = tan(rad(angle_deg))

    print(f"\n--- Symbolic Results for angle {angle_deg}° ---")
    print(f"sin({angle_deg}) = {pretty(sin_val)} ≈ {N(sin_val):.4f}")
    print(f"cos({angle_deg}) = {pretty(cos_val)} ≈ {N(cos_val):.4f}")
    if isclose(N(cos_val), 0.0, abs_tol=1e-10):
        print("tan is undefined (cos = 0)")
    else:
        print(f"tan({angle_deg}) = {pretty(tan_val)} ≈ {N(tan_val):.4f}")

def calculate_from_sides(a=None, b=None, c=None):
    """
    Calculates sin, cos, and tan based on known sides of a right triangle.
    Accepts uncertain values using the 'uncertainties' module if needed.
    """
    if c is None:
        if a is not None and b is not None:
            c = sqrt(a**2 + b**2)
        else:
            print("Insufficient sides provided.")
            return

    print("\n--- Results from triangle sides ---")
    if a is not None and c is not None:
        sin_val = a / c
        print(f"sin(v) = {sin_val:.4f}")
    if b is not None and c is not None:
        cos_val = b / c
        print(f"cos(v) = {cos_val:.4f}")
    if a is not None and b is not None:
        tan_val = a / b
        print(f"tan(v) = {tan_val:.4f}")