import geometry as gm
import igl
import jax.numpy as jnp
from os import path
import numpy as np
import mayavi.mlab as mlab

data_dir = path.join(path.dirname(__file__), "data")
v, f = igl.read_triangle_mesh(path.join(data_dir, "usb_male.obj"))
b_min = np.min(v, axis=0)
b_max = np.max(v, axis=0)
b_lens = b_max - b_min
bbox = np.array([b_min - 0.1*b_lens, b_max + 0.1*b_lens])
sampler = gm.MeshSampler(v, f, bbox, ratio=0.0, std=0.05*b_lens.min())

x_sample = sampler.sample_points(10000)
sdf_sample = sampler.sdf(x_sample)

fig = mlab.figure()
fig.scene.background = (1,1,1)
mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f, color=(0.2,0.8,0.5))
# mlab.points3d(x_sample[:,0], x_sample[:,1], x_sample[:,2], sdf_sample, colormap="copper", mode="sphere", scale_factor=0.0002)
mlab.points3d(x_sample[:,0], x_sample[:,1], x_sample[:,2], sdf_sample, colormap="jet", mode="sphere", scale_mode='none', scale_factor=0.0002)
mlab.show()

