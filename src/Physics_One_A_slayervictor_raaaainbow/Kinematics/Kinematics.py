def test():
    """A function for plotting the boundary of a 2D area

    Args:
        param_func (MatrixBase,): A matrix with functions [x(u,v), y(u,v)] determining x and y-coordinates of area to be plotted.
        U_lim (Tuple[Symbol, float, float]): A tuple with variable and its limits either as list or two seperate values. Example as (u, u_min, u_max).
        V_lim (Tuple[Symbol, float, float]): A tuple with variable and its limits either as list or two seperate values. Example as (v, v_min, v_max).
        rendering_kw (dict, optional): A dictionary forwarded to dtuplot.plot(), see SPB docs for reference.
        color (str, optional): A string to set color boundary with. With no argument color = 'blue'.
        show (bool, optional): Boolean, if 'True': show plot, other just return object without plotting. Defaults to 'True'.


    Returns:
        Plot: A SPB-plot object
    """
    pass