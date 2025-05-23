from Lectures.dependencies import *
def evalf(n):
    return N(n)

def speed_converter(value, from_unit='m/s', to_unit='km/h'):
   """
   Convert between m/s and km/h.
   
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