import jax.numpy as jnp
from jax import vmap
from interpax import *
import matplotlib.pyplot as plt

# Ground truth SDF
# Circle of radius 3
f = lambda x : jnp.linalg.norm(x) - 3

# Initialize interpolator
# Grid points
num_grid_pts = [10,10]
g1s = jnp.linspace(-10.0, 10.0, num_grid_pts[0])
g2s = jnp.linspace(-10.0, 10.0, num_grid_pts[1])
g1_sample, g2_sample = jnp.meshgrid(g1s, g2s)
# g1_sample = g1_sample.reshape(-1)
# g2_sample = g2_sample.reshape(-1)
g_sample = jnp.stack([g1_sample, g2_sample], axis=-1)
interp = Interpolator2D(
    g1s,
    g2s,
    vmap(vmap(f))(g_sample),
    method="cubic2",
)

# Sample the interpolation
x1s = jnp.linspace(-10.0, 10.0, 1000)
x2s = jnp.linspace(-10.0, 10.0, 1000)
x1_sample, x2_sample = jnp.meshgrid(x1s, x2s)
# x_sample = jnp.stack([x1_sample, x2_sample], axis=-1)
f_sample = interp(x1_sample.reshape(-1), x2_sample.reshape(-1)).reshape(1000,1000)

f_true = vmap(vmap(f))(jnp.stack([x1_sample, x2_sample], axis=-1))
error = jnp.abs(f_sample - f_true)

plt.figure(1)
plt.pcolormesh(x1_sample, x2_sample, f_sample)

plt.figure(2)
plt.pcolormesh(x1_sample, x2_sample, error)
plt.show()
