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

def net_force_from_velocity_change(mass, dv, dt):
    """
    Calculates the net force from a change in velocity over time:
        F = m * (dv / dt)
    
    Parameters:
        mass: mass of the object [kg]
        dv: change in velocity [m/s]
        dt: time interval [s]
    
    Returns:
        Net force [N]
    """
    m = format_input(mass)
    dv = format_input(dv)
    dt = format_input(dt)
    return m * (dv / dt)


def acceleration_of_two_blocks(F, mA, mB, g=gravity()):
    """
    Calculates acceleration of both blocks as a system:
    a = (F - (mA + mB) * g) / (mA + mB)
    """
    F = format_input(F)
    mA = format_input(mA)
    mB = format_input(mB)
    g = format_input(g)
    total_mass = mA + mB
    return (F - total_mass * g) / total_mass

def tension_in_rope(mB, acceleration, g=gravity()):
    """
    Calculates the tension in the rope acting on mB:
    T = mB * (a + g)
    """
    mB = format_input(mB)
    a = format_input(acceleration)
    g = format_input(g)
    return mB * (a + g)

def inclined_plane_acceleration(mass, force, theta_deg, g=gravity()):
    """
    Returns the acceleration of a block pushed horizontally up a frictionless incline.

    Formula:
        a = (F * cos(theta))/m - g * sin(theta)
    """
    m = format_input(mass)
    F = format_input(force)
    theta = degree_to_radian(theta_deg)
    g = format_input(g)

    return (F * cos(theta)) / m - g * sin(theta)


def normal_force_on_incline(mass, force, theta_deg, g=gravity()):
    """
    Returns the normal force on the block on a frictionless incline.

    Formula:
        N = m * g * cos(theta) + F * sin(theta)
    """
    m = format_input(mass)
    F = format_input(force)
    theta = degree_to_radian(theta_deg)
    g = format_input(g)

    return m * g * cos(theta) + F * sin(theta)


def three_block_acceleration(mA, mB, mC, force):
    """
    Returns total acceleration of three-block system pulled by force F.
    """
    mA = format_input(mA)
    mB = format_input(mB)
    mC = format_input(mC)
    F = format_input(force)

    total_mass = mA + mB + mC
    return F / total_mass


def tension_bc(mC, acceleration):
    """
    Tension acting on block C (pushed by block B).
    """
    mC = format_input(mC)
    a = format_input(acceleration)
    return mC * a


def tension_ab(mB, mC, acceleration):
    """
    Tension between A and B (acts on combined B+C).
    """
    mB = format_input(mB)
    mC = format_input(mC)
    a = format_input(acceleration)
    return (mB + mC) * a

def unstretched_length_of_spring(mass, radius, period, spring_constant):
    """
    Calculates the unstretched length L0 of a spring during circular motion.
    
    L0 = R - (m * v²) / (k * R)
    where v = 2πR / T
    """
    m = format_input(mass)
    R = format_input(radius)
    T = format_input(period)
    k = format_input(spring_constant)

    v = (2 * np.pi * R) / T
    return R - (m * v**2) / (k * R)