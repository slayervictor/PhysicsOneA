from PhysicsOneA.dependencies import *
from uncertainties import ufloat

def solve_suvat(s=None, u=None, v=None, a=None, t=None):
    """
    Solves the SUVAT equations for uniformly accelerated motion.
    
    Accepts both regular numbers and ufloat objects with uncertainties.
    Returns results with proper uncertainty propagation when applicable.

    Parameters:
        s, u, v, a, t: float, ufloat, or None
            - s: displacement (m)
            - u: initial velocity (m/s)  
            - v: final velocity (m/s)
            - a: acceleration (m/s²)
            - t: time (s)

    Returns:
        dict: All five variables with uncertainties preserved if present

    Example:
        >>> solve_suvat(u=ufloat(0, 0.1), a=2, t=3)
        {'s': 9.0+/-0.90, 'u': 0.0+/-0.10, 'v': 6.0+/-0.30, 'a': 2.0, 't': 3.0}
    """
    
    def normalize_input(value):
        """Convert input to ufloat for consistent processing."""
        if value is None:
            return None
        elif hasattr(value, 'nominal_value'):
            return value  # Already a ufloat
        else:
            return ufloat(float(value), 0)  # Regular number with zero uncertainty
    
    def get_nominal(value):
        """Extract nominal value for SymPy computation."""
        if value is None:
            return None
        elif hasattr(value, 'nominal_value'):
            return value.nominal_value
        else:
            return float(value)
    
    def has_uncertainties(*values):
        """Check if any input has non-zero uncertainty."""
        for val in values:
            if val is not None and hasattr(val, 'std_dev') and val.std_dev > 0:
                return True
        return False
    
    def maybe_simplify(value, preserve_uncertainty):
        """Convert to float if no uncertainties were in the inputs."""
        if not preserve_uncertainty and hasattr(value, 'nominal_value'):
            return float(value.nominal_value)
        return value
    
    # Normalize all inputs
    inputs = [s, u, v, a, t]
    s_norm, u_norm, v_norm, a_norm, t_norm = [normalize_input(x) for x in inputs]
    
    # Check if we need to preserve uncertainties
    preserve_uncertainty = has_uncertainties(s, u, v, a, t)
    
    # Count known values
    known = [x for x in [s_norm, u_norm, v_norm, a_norm, t_norm] if x is not None]
    if len(known) != 3:
        raise ValueError("Exactly 3 values must be provided")
    
    # Use SymPy to solve with nominal values (original logic)
    s_sym, u_sym, v_sym, a_sym, t_sym = symbols('s u v a t', real=True)
    
    eq1 = Eq(v_sym, u_sym + a_sym * t_sym)  
    eq2 = Eq(s_sym, u_sym * t_sym + a_sym * t_sym**2 / 2)  
    eq3 = Eq(v_sym**2, u_sym**2 + 2 * a_sym * s_sym)  
    
    equations = [eq1, eq2, eq3]
    variables = [s_sym, u_sym, v_sym, a_sym, t_sym]
    
    # Substitute known nominal values
    substitutions = {}
    if s_norm is not None: substitutions[s_sym] = get_nominal(s_norm)
    if u_norm is not None: substitutions[u_sym] = get_nominal(u_norm)
    if v_norm is not None: substitutions[v_sym] = get_nominal(v_norm)
    if a_norm is not None: substitutions[a_sym] = get_nominal(a_norm)
    if t_norm is not None: substitutions[t_sym] = get_nominal(t_norm)
    
    eqs_substituted = [eq.subs(substitutions) for eq in equations]
    unknowns = [var for var in variables if var not in substitutions]
    
    try:
        solutions = solve(eqs_substituted, unknowns, dict=True)
        
        if not solutions:
            raise ValueError("No solution found for given values")
        
        solution = solutions[0]
        if len(solutions) > 1 and t_sym in unknowns:
            # Choose positive time solution
            for sol in solutions:
                if t_sym in sol and sol[t_sym] >= 0:
                    solution = sol
                    break
        
        # Build result using uncertainty-aware computation
        result = {}
        
        # For known values, use original inputs (preserving uncertainties)
        if s_norm is not None:
            result['s'] = maybe_simplify(s_norm, preserve_uncertainty)
        else:
            # Compute s with uncertainty propagation
            s_val = u_norm * t_norm + 0.5 * a_norm * t_norm**2
            result['s'] = maybe_simplify(s_val, preserve_uncertainty)
            
        if u_norm is not None:
            result['u'] = maybe_simplify(u_norm, preserve_uncertainty)
        else:
            # u = v - at
            u_val = v_norm - a_norm * t_norm
            result['u'] = maybe_simplify(u_val, preserve_uncertainty)
            
        if v_norm is not None:
            result['v'] = maybe_simplify(v_norm, preserve_uncertainty)
        else:
            # v = u + at
            v_val = u_norm + a_norm * t_norm
            result['v'] = maybe_simplify(v_val, preserve_uncertainty)
            
        if a_norm is not None:
            result['a'] = maybe_simplify(a_norm, preserve_uncertainty)
        else:
            # a = (v - u) / t
            a_val = (v_norm - u_norm) / t_norm
            result['a'] = maybe_simplify(a_val, preserve_uncertainty)
            
        if t_norm is not None:
            result['t'] = maybe_simplify(t_norm, preserve_uncertainty)
        else:
            # t = (v - u) / a
            t_val = (v_norm - u_norm) / a_norm
            result['t'] = maybe_simplify(t_val, preserve_uncertainty)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Could not solve equations: {e}")
