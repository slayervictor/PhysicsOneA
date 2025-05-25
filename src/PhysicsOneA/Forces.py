from PhysicsOneA.dependencies import *
from PhysicsOneA.helpers import *

def spring_force(k, x, x0=0):
    """
    Hooke's law: F = -k * (x - x0)
    """
    k = format_input(k)
    x = format_input(x)
    x0 = format_input(x0)
    return -k * (x - x0)

def kinetic_friction(mu_k, normal_force):
    """
    f_k = μ_k * N
    """
    mu_k = format_input(mu_k)
    normal_force = format_input(normal_force)
    return mu_k * normal_force

def static_friction_max(mu_s, normal_force):
    """
    f_s ≤ μ_s * N
    """
    mu_s = format_input(mu_s)
    normal_force = format_input(normal_force)
    return mu_s * normal_force
def general_drag(b, v, n=1):
    """
    F_d = b * v^n
    """
    b = format_input(b)
    v = format_input(v)
    return b * v**n

def drag_force(D, v):
    """
    f = D * v^2
    """
    D = format_input(D)
    v = format_input(v)
    return D * v**2

def drag_coefficient(rho, C_D, A):
    """
    D = 1/2 * ρ * C_D * A
    """
    rho = format_input(rho)
    C_D = format_input(C_D)
    A = format_input(A)
    return 0.5 * rho * C_D * A

def stokes_drag(radius, viscosity, velocity):
    """
    F_s = 6π * η * r * v
    """
    eta = format_input(viscosity)
    r = format_input(radius)
    v = format_input(velocity)
    return 6 * np.pi * eta * r * v

def net_force(mass, acceleration):
    """
    Newtons 2. lov: F = m * a
    """
    m = format_input(mass)
    a = format_input(acceleration)
    return m * a

