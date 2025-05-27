from PhysicsOneA.dependencies import *
from PhysicsOneA.helpers import *

def rotational_kinetic_energy(I, omega):
    """
    Calculates rotational kinetic energy.

    Formula:
        K_rot = 1/2 * I * omega^2

    Parameters:
        I (float or UFloat): Moment of inertia (kg*m^2)
        omega (float or UFloat): Angular velocity in rad/s

    Returns:
        UFloat: Rotational kinetic energy in joules
    """
    I = format_input(I)
    omega = format_input(omega)
    return 0.5 * I * omega**2


def combined_kinetic_energy(I, omega, mass, v_cm):
    """
    Calculates total kinetic energy of a rotating and translating object.

    Formula:
        K = 1/2 * I * omega^2 + 1/2 * M * v_cm^2

    Parameters:
        I (float or UFloat): Moment of inertia (kg*m^2)
        omega (float or UFloat): Angular velocity (rad/s)
        mass (float or UFloat): Mass (kg)
        v_cm (float or UFloat): Velocity of center of mass (m/s)

    Returns:
        UFloat: Total kinetic energy (J)
    """
    K_rot = rotational_kinetic_energy(I, omega)
    m = format_input(mass)
    v = format_input(v_cm)
    K_trans = 0.5 * m * v**2
    return K_rot + K_trans


def parallel_axis(I_cm, mass, distance):
    """
    Applies the parallel axis theorem.

    Formula:
        I = I_cm + M * d^2

    Parameters:
        I_cm (float or UFloat): Moment of inertia through CM (kg*m^2)
        mass (float or UFloat): Mass (kg)
        distance (float or UFloat): Distance from new axis to CM (m)

    Returns:
        UFloat: Adjusted moment of inertia (kg*m^2)
    """
    I_cm = format_input(I_cm)
    m = format_input(mass)
    d = format_input(distance)
    return I_cm + m * d**2


def rotational_energy_conservation(K1_rot, K1_trans, U1, W_other, K2_trans, U2):
    """
    Computes final rotational kinetic energy using energy conservation.

    Formula:
        K1_rot + K1_trans + U1 + W_other = K2_rot + K2_trans + U2

    Parameters:
        K1_rot, K1_trans, U1, W_other, K2_trans, U2: all energy terms (float or UFloat)

    Returns:
        UFloat: K2_rot (final rotational kinetic energy)
    """
    terms = [K1_rot, K1_trans, U1, W_other, K2_trans, U2]
    values = [format_input(x) for x in terms]
    return sum(values[:4]) - sum(values[4:])


# --- Inertia formulas for common shapes ---

def inertia_solid_cylinder(mass, radius):
    """
    Moment of inertia of a solid cylinder about its central axis.
    I = 1/2 * M * R^2
    """
    m = format_input(mass)
    R = format_input(radius)
    return 0.5 * m * R**2


def inertia_hollow_cylinder(mass, R1, R2):
    """
    Moment of inertia of a hollow cylinder.
    I = 1/2 * M * (R1^2 + R2^2)
    """
    m = format_input(mass)
    R1 = format_input(R1)
    R2 = format_input(R2)
    return 0.5 * m * (R1**2 + R2**2)


def inertia_hoop(mass, radius):
    """
    Moment of inertia of a hoop about central axis.
    I = M * R^2
    """
    m = format_input(mass)
    R = format_input(radius)
    return m * R**2


def inertia_sphere(mass, radius):
    """
    Moment of inertia of a solid sphere.
    I = 2/5 * M * R^2
    """
    m = format_input(mass)
    R = format_input(radius)
    return (2/5) * m * R**2


def inertia_spherical_shell(mass, radius):
    """
    Moment of inertia of a thin spherical shell.
    I = 2/3 * M * R^2
    """
    m = format_input(mass)
    R = format_input(radius)
    return (2/3) * m * R**2


def inertia_thin_rod_center(mass, length):
    """
    Moment of inertia of a thin rod through its center.
    I = 1/12 * M * L^2
    """
    m = format_input(mass)
    L = format_input(length)
    return (1/12) * m * L**2


def inertia_thin_rod_end(mass, length):
    """
    Moment of inertia of a thin rod through one end.
    I = 1/3 * M * L^2
    """
    m = format_input(mass)
    L = format_input(length)
    return (1/3) * m * L**2