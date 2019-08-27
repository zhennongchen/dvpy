import math
import numpy as np

def decompositeQ(Q):
    a = math.sqrt(1 - Q[0]**2 - Q[1]**2 - Q[-1]**2)
    a_rad = math.acos(a) * 2
    a = a_rad / math.pi * 180
    #print('\nangle',a)
    sin = math.sin(a_rad/2)
    [ux,uy,uz] = [Q[0] / sin, Q[1] / sin, Q[-1] / sin]
    #print('axis',[ux,uy,uz])
    u = np.array([ux,uy,uz])
    
    return np.array([a_rad]),u