import os
from os import path
import igl
import numpy as np


v, f = igl.read_triangle_mesh(path.join(path.dirname(__file__), "usb_male.obj"))

## Print the vertices and faces matrices 
print("Vertices: ", len(v))
print("Faces: ", len(f))

# Get a point on the surface
f_id = np.random.choice(len(f))
f_vs = f[f_id,:]
x = 


# Let's try to get signed distance
