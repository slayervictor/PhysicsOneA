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
        
        # Build result dictionary with computed values
        result = {}
        
        # Use SymPy solution values, but reconstruct with uncertainty propagation
        s_final = ufloat(float(solution.get(s_sym, get_nominal(s_norm))), 0) if s_norm is None else s_norm
        u_final = ufloat(float(solution.get(u_sym, get_nominal(u_norm))), 0) if u_norm is None else u_norm  
        v_final = ufloat(float(solution.get(v_sym, get_nominal(v_norm))), 0) if v_norm is None else v_norm
        a_final = ufloat(float(solution.get(a_sym, get_nominal(a_norm))), 0) if a_norm is None else a_norm
        t_final = ufloat(float(solution.get(t_sym, get_nominal(t_norm))), 0) if t_norm is None else t_norm
        
        # Recompute unknown values using uncertainty-aware equations to get proper error propagation
        if s_norm is None and u_final is not None and a_final is not None and t_final is not None:
            s_final = u_final * t_final + 0.5 * a_final * t_final**2
        if u_norm is None and v_final is not None and a_final is not None and t_final is not None:
            u_final = v_final - a_final * t_final
        if v_norm is None and u_final is not None and a_final is not None and t_final is not None:
            v_final = u_final + a_final * t_final
        if a_norm is None and v_final is not None and u_final is not None and t_final is not None:
            a_final = (v_final - u_final) / t_final
        if t_norm is None and v_final is not None and u_final is not None and a_final is not None:
            t_final = (v_final - u_final) / a_final
            
        result['s'] = maybe_simplify(s_final, preserve_uncertainty)
        result['u'] = maybe_simplify(u_final, preserve_uncertainty)
        result['v'] = maybe_simplify(v_final, preserve_uncertainty)
        result['a'] = maybe_simplify(a_final, preserve_uncertainty)
        result['t'] = maybe_simplify(t_final, preserve_uncertainty)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Could not solve equations: {e}")
