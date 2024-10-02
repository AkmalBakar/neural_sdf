import jax.numpy as jnp

def circle_sdf(x):
    return jnp.linalg.norm(x) - 3